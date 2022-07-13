import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
# turn off TF32 for higher accuracy
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

import time
import taichi as ti

######################################################################################################
# From https://github.com/BlinkDL/RWKV-CUDA
# On GTX1070 mobile:
# pytorch = fwd 94ms bwd 529ms
# CUDA kernel v0 = fwd 45ms bwd 84ms (simple)
# CUDA kernel v1 = fwd 17ms bwd 43ms (shared memory)
# CUDA kernel v2 = fwd 13ms bwd 31ms (float4)
# CUDA kernel v3 = fwd 3.4ms bwd 23ms (B-group)
######################################################################################################

CUDA_KERNEL_VERSION = 3 # CUDA kernel version = 0,1,2


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

######################################################################################################
# The formula:
# w.shape = (C, T)
# k.shape = (B, C, T)
# out.shape = (B, C, T)
# out[b][c][t] = sum_u{ w[c][(T-1)-(t-u)] * k[b][c][u] }
######################################################################################################


def RUN_FORMULA_VERY_SLOW(w, k, B, C, T, eps):
    # this is the formula (very slow)
    out = torch.empty((B, C, T), device='cuda')
    for b in range(B):
        for c in range(C):
            for t in range(T):
                s = eps
                for u in range(0, t+1):
                    s += w[c][(T-1)-(t-u)] * k[b][c][u]
                out[b][c][t] = s
    return out


def RUN_PYTORCH(w, k, B, C, T, eps):
    # this shall equal the formula
    return F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(k), w.unsqueeze(1), groups=C) + eps

@ti.kernel
def timex_taichi_forward_baseline(out: ti.types.ndarray(field_dim=3),
        w: ti.types.ndarray(field_dim=2),
        k: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32, eps: ti.f32):
    for b, c, t in out:
        s = eps
        for u in range(0, t+1):
            s += w[c, (T-1)-(t-u)] * k[b, c, u]
        out[b, c, t] = s

ti_vecf = ti.types.vector(4, dtype=float)
@ti.kernel
def timex_taichi_forward_group_t(out: ti.types.ndarray(field_dim=3),
        w: ti.types.ndarray(field_dim=2),
        k: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32, eps: ti.f32):
    for b, c, t in ti.ndrange(B, C, T // 4):
        # Group t with factor 4
        t_block = t * 4
        s_mat = ti_vecf((eps, eps, eps, eps))
        for u in range(0, t_block+1):
            for i in ti.static(range(4)):
                s_mat[i] += w[c, (T-1)-(t_block-u+i)] * k[b, c, u]
        # Compute the remaining triangle.
        for j in ti.static(range(1, 4)):
            for i in ti.static(range(j)):
                s_mat[j] += w[c, T - j + i] * k[b, c, t_block + 1 + i]
        for i_t in ti.static(range(4)):
            out[b, c, t_block + i_t] = s_mat[i_t]

t_group = 6
b_group = 8
ti_matf = ti.types.matrix(b_group, t_group, dtype=float)
@ti.kernel
def timex_taichi_forward(
        out: ti.types.ndarray(field_dim=3),
        w: ti.types.ndarray(field_dim=2),
        k: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32, eps: ti.f32):
    for b, c, t in ti.ndrange(B // b_group, C, T // t_group):
        # Group both b and t with factor 4
        t_block = t * t_group
        s_mat = ti_matf(((eps,) * t_group,) * b_group)
        for u in range(0, t_block+1):
            for bi in ti.static(range(b_group)):
                k_val = k[b * b_group + bi, c, u]
                for i in ti.static(range(t_group)):
                    s_mat[bi, i] += w[c, (T-1) - (t_block - (u - i))] * k_val
        # Compute the remaining triangle in the thread group.
        for bi in ti.static(range(b_group)):
            for j in ti.static(range(1, t_group)):
                for i in ti.static(range(j)):
                    s_mat[bi, j] += w[c, T - j + i] * k[b * b_group + bi, c, t_block + 1 + i]
            for i_t in ti.static(range(t_group)):
                out[b * b_group + bi, c, t_block + i_t] = s_mat[bi, i_t]

@ti.kernel
def timex_taichi_backward_baseline(
        w: ti.types.ndarray(field_dim=2),
        k: ti.types.ndarray(field_dim=3),
        gwk: ti.types.ndarray(field_dim=3),
        gw: ti.types.ndarray(field_dim=3),
        gk: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32):
    for b, c, t in gwk:
        s = 0.0
        for u in range(0, t+1):
            s += gwk[b, c, (T-1)-(t-u)] * k[b, c, u]
        gw[b, c, t] = s
        s = 0.0
        for u in range(t, T):
            s += gwk[b, c, (T-1)+(t-u)] * w[c, u]
        gk[b, c, t] = s

t_group = 6
ti_vecf = ti.types.vector(t_group, dtype=float)
@ti.kernel
def timex_taichi_backward_group_t(
        w: ti.types.ndarray(field_dim=2),
        k: ti.types.ndarray(field_dim=3),
        gwk: ti.types.ndarray(field_dim=3),
        gw: ti.types.ndarray(field_dim=3),
        gk: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32):
    for b, c, t_block in ti.ndrange(B, C, T // t_group):
        t = t_block * t_group
        s = ti_vecf((0.0) * t_group)
        for u in range(0, t+1):
            for i in ti.static(range(0, t_group)):
                s[i] += gwk[b, c, (T-1)-(t+i-u)] * k[b, c, u]
        # The last triangle is specialized
        # u is replaced with t+1+j
        for i in ti.static(range(1, t_group)):
            for j in ti.static(range(i)):
                s[i] += gwk[b, c, T-i+j] * k[b, c, t+1+j]
        # write out
        for i in ti.static(range(t_group)):
            gw[b, c, t+i] = s[i]

        s = ti_vecf((0.0) * t_group)
        # The first triangle is specialized
        # t' = t + i
        # u' = t' + j 
        for i in ti.static(range(0, t_group-1)):
            for j in ti.static(range(i, t_group-1)):
                s[i] += gwk[b, c, T+i-j-1] * w[c, t+j]

        for u in range(t+t_group-1, T):
            for i in ti.static(range(0, t_group)):
                s[i] += gwk[b, c, (T-1)+(t+i-u)] * w[c, u]
        # write out
        for i in ti.static(range(t_group)):
            gk[b, c, t+i] = s[i]

bwd_t = 6
bwd_b = 4
ti_back_matf = ti.types.matrix(bwd_b, bwd_t, dtype=float)
@ti.kernel
def timex_taichi_backward(
        w: ti.types.ndarray(field_dim=2),
        k: ti.types.ndarray(field_dim=3),
        gwk: ti.types.ndarray(field_dim=3),
        gw: ti.types.ndarray(field_dim=3),
        gk: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32):
    for b_block, c, t_block in ti.ndrange(B // bwd_b, C, T // bwd_t):
        t = t_block * bwd_t
        b = b_block * bwd_b
        s = ti_back_matf(((0.0,) * bwd_t,)*bwd_b)
        for bi in ti.static(range(0, bwd_b)):
            for u in range(0, t+1):
                for i in ti.static(range(0, bwd_t)):
                    s[bi, i] += gwk[b + bi, c, (T-1)-(t+i-u)] * k[b + bi, c, u]
        # The last triangle is specialized
        # u is replaced with t+1+j
        for bi in ti.static(range(0, bwd_b)):
            for i in ti.static(range(1, bwd_t)):
                for j in ti.static(range(i)):
                    s[bi, i] += gwk[b + bi, c, T-i+j] * k[b + bi, c, t+1+j]
        # write out
        for bi in ti.static(range(0, bwd_b)):
            for i in ti.static(range(bwd_t)):
                gw[b + bi, c, t+i] = s[bi, i]

        s = ti_back_matf(((0.0,) * bwd_t,)*bwd_b)
        # The first triangle is specialized
        # t' = t + i
        # u' = t' + j 
        for bi in ti.static(range(0, bwd_b)):
            for i in ti.static(range(0, bwd_t-1)):
                for j in ti.static(range(i, bwd_t-1)):
                    s[bi, i] += gwk[b+bi, c, T+i-j-1] * w[c, t+j]

        for bi in ti.static(range(0, bwd_b)):
            for u in range(t+bwd_t-1, T):
                for i in ti.static(range(0, bwd_t)):
                    s[bi, i] += gwk[b+bi, c, (T-1)+(t+i-u)] * w[c, u]
        # write out
        for bi in ti.static(range(0, bwd_b)):
            for i in ti.static(range(bwd_t)):
                gk[b+bi, c, t+i] = s[bi, i]


######################################################################################################
# Load the CUDA kernel
######################################################################################################

T_MAX = 768
B_GROUP_FORWARD = 4
B_GROUP_BACKWARD = 2

timex_cuda = load(name="timex", sources=["cuda/timex_op.cpp", "cuda/timex_cuda_v" + str(CUDA_KERNEL_VERSION) + ".cu"],
                  verbose=True, extra_cuda_cflags=['--use_fast_math', '--extra-device-vectorization', f'-DTmax={T_MAX}', f'-DBF={B_GROUP_FORWARD}', f'-DBB={B_GROUP_BACKWARD}'])

# we call it the "TimeX" operator because it's used for time-mixing in my RWKV language model
class TimeX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, k, B, C, T, eps):
        ctx.B = B
        ctx.C = C
        ctx.T = T
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0, "require T % 4 == 0 and T <= T_MAX and B % B_GROUP_* == 0"
        w = w.contiguous()
        k = k.contiguous()
        ctx.save_for_backward(w, k)
        wk = torch.empty((B, C, T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_cuda.forward(w, k, wk, eps, B, C, T)

        return wk

    @staticmethod
    def backward(ctx, gwk):
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0, "require T % 4 == 0 and T <= T_MAX and B % B_GROUP_* == 0"
        w, k = ctx.saved_tensors
        gw = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        gk = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_cuda.backward(w, k, gwk.contiguous(), gw,
                            gk, ctx.B, ctx.C, ctx.T)
        # actually pytorch will do gw.sum(dim=0) but we will do it anyway just to be safe
        return (gw.sum(dim=0), gk, None, None, None, None)

class TimeX_Taichi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, k, B, C, T, eps):
        ctx.B = B
        ctx.C = C
        ctx.T = T
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0, "require T % 4 == 0 and T <= T_MAX and B % B_GROUP_* == 0"
        w = w.contiguous()
        k = k.contiguous()
        ctx.save_for_backward(w, k)
        wk = torch.empty((B, C, T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_taichi_forward(wk, w, k, B, C, T, eps)
        ti.sync()
        return wk

    @staticmethod
    def backward(ctx, gwk):
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0, "require T % 4 == 0 and T <= T_MAX and B % B_GROUP_* == 0"
        w, k = ctx.saved_tensors
        gw = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        gk = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_taichi_backward(w, k, gwk.contiguous(), gw,
                            gk, ctx.B, ctx.C, ctx.T)
        ti.sync()
        # actually pytorch will do gw.sum(dim=0) but we will do it anyway just to be safe
        return (gw.sum(dim=0), gk, None, None, None, None)

def RUN_CUDA(w, k, B, C, T, eps):
    return TimeX.apply(w.cuda(), k.cuda(), B, C, T, eps)

def RUN_TAICHI(w, k, B, C, T, eps):
    return TimeX_Taichi.apply(w.cuda(), k.cuda(), B, C, T, eps)



######################################################################################################
# Check correctness & speed benchmark
######################################################################################################

def CHECK_PYTORCH():
    B = 3
    C = 5
    T = 11
    eps = 0.1

    set_seed(42)
    w = torch.rand(C, T, requires_grad=True, device='cuda')
    k = torch.rand(B, C, T, requires_grad=True, device='cuda')

    r0 = RUN_FORMULA_VERY_SLOW(w, k, B, C, T, eps)
    r1 = RUN_PYTORCH(w, k, B, C, T, eps)

    print('--> pytorch correct =', torch.allclose(r0, r1),
          ', err ratio =', get_err_ratio(r0, r1))

def CHECK_ALL(silent=False):
    B = 32
    C = 768
    T = 768
    eps = 0.1

    set_seed(42)
    w = torch.rand(C, T, requires_grad=True, device='cuda')
    k = torch.rand(B, C, T, requires_grad=True, device='cuda')

    # check forward

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        r1 = RUN_PYTORCH(w, k, B, C, T, eps)
    if not silent:
        print('pytorch forward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        r2 = RUN_CUDA(w, k, B, C, T, eps)
    if not silent:
        print('CUDA forward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))

    # check backward
    # a strange loss for better verification
    loss1 = ((r1 * r1) - torch.tanh(r1)).sum()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        loss1.backward()
    if not silent:
        print('pytorch backward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))
    gw1 = w.grad.data.clone()
    gk1 = k.grad.data.clone()

    w.grad.data.zero_()
    k.grad.data.zero_()

    loss2 = ((r2 * r2) - torch.tanh(r2)).sum()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        loss2.backward()
    if not silent:
        print('CUDA backward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))
    gw2 = w.grad.data.clone()
    gk2 = k.grad.data.clone()

    print('--> bwd gradW correct =', torch.allclose(gw1, gw2),
          ', err ratio =', get_err_ratio(gw1, gw2))
    print('--> bwd gradK correct =', torch.allclose(gk1, gk2),
          ', err ratio =', get_err_ratio(gk1, gk2))

    w.grad.data.zero_()
    k.grad.data.zero_()

    # Check Taichi
    ti.init(arch=ti.cuda, kernel_profiler=True)
    # Taichi
    r3 = RUN_TAICHI(w, k, B, C, T, eps)
    loss3 = ((r3 * r3) - torch.tanh(r3)).sum()
    loss3.backward()
    w.grad.data.zero_()
    k.grad.data.zero_()
    ti.profiler.clear_kernel_profiler_info()
    r3 = RUN_TAICHI(w, k, B, C, T, eps)
    ti.sync()

    print('--> Taichi fwd correct =', torch.allclose(r1, r3),
         ', err ratio =', get_err_ratio(r1, r3))
    loss3 = ((r3 * r3) - torch.tanh(r3)).sum()
    loss3.backward()
    if not silent:
        ti.profiler.print_kernel_profiler_info('trace')
    gw3 = w.grad.data.clone()
    gk3 = k.grad.data.clone()

    print('--> bwd gradW correct =', torch.allclose(gw1, gw3),
         ', err ratio =', get_err_ratio(gw1, gw3))
    print('--> bwd gradK correct =', torch.allclose(gk1, gk3),
         ', err ratio =', get_err_ratio(gk1, gk3))



if __name__ == "__main__":
    print('\n\nCUDA benchmark...')
    CHECK_ALL(silent=False)  # benchmark
