import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pdb
import taichi as ti
ti.init()

np.set_printoptions(precision=4, suppress=True, linewidth=200)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def RUN_FORMULA_VERY_SLOW(w, k, B, C, T, eps):
    # this is the formula (very slow)
    out = torch.empty((B, C, T), device='cpu')
    for b in range(B):
        for c in range(C):
            for t in range(T):
                s = eps
                for u in range(t-T+1, t+1):
                    s += w[c][0][(T-1)-(t-u)] * k[b][c][u+T-1]
                out[b][c][t] = s
    return out


def RUN_PYTORCH(w, k, B, C, T, eps):
    # this shall equal the formula
    return F.conv1d(k, w, groups=C) + eps

@ti.kernel
def timex_taichi_forward_baseline(out: ti.types.ndarray(field_dim=3),
        w: ti.types.ndarray(field_dim=3),
        k: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32, eps: ti.f32):
    for b, c, t in out:
        s = eps
        for u in range(t-T+1, t+1):
            s += w[c, 0, (T-1)-(t-u)] * k[b, c, u+T-1]
        out[b, c, t] = s


def CHECK_PYTORCH():
    B = 3
    C = 5
    T = 11
    eps = 0.1

    set_seed(42)
    w = torch.rand(C, T, requires_grad=True, device='cpu')
    w = w.unsqueeze(1)
    k = torch.rand(B, C, T, requires_grad=True, device='cpu')
    k = nn.ZeroPad2d((T-1, 0, 0, 0))(k)

    r0 = RUN_FORMULA_VERY_SLOW(w, k, B, C, T, eps)
    r1 = RUN_PYTORCH(w, k, B, C, T, eps)

    print('--> pytorch correct =', torch.allclose(r0, r1),
          ', err ratio =', get_err_ratio(r0, r1))

    out = torch.empty((B, C, T), device='cpu')
    timex_taichi_forward_baseline(out, w, k, B, C, T, eps)
    print('--> taichi correct=', torch.allclose(r0, out),
        ', err ratio =', get_err_ratio(r0, out))

CHECK_PYTORCH()
