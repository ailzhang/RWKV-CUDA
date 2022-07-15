#include <stdio.h>

// require T % 4 == 0 and T <= Tmax (passed by compiler)

#define F4(A, B) ((float4 *)(A))[(B) >> 2]

template <typename F>
__global__ void kernel_forward(const F *__restrict__ const __w, const F *__restrict__ const __k, F *__restrict__ const x,
                               const F eps, const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int t = threadIdx.x << 2;

    const int warp_id = threadIdx.x % 32;

    const int ti = t + T * i;
    const int tj = T * (B * C) / BF;

    const F *__restrict__ const www = __w + (i % C) * T + (T - 4) - t;
    float4 s[BF];
#pragma unroll
    for (int j = 0; j < BF; j++) {
        s[j] = {eps, eps, eps, eps};
    }

    for (int u = 0; u <= t; u++) {
#pragma unroll
        for (int j = 0; j < BF; j++) { 
            F k_val = __k[u + ti + tj * j - t];
            s[j].x += www[u + 3] * k_val;
            s[j].y += www[u + 2] * k_val;
            s[j].z += www[u + 1] * k_val;
            s[j].w += www[u + 0] * k_val;
        }
    }
#pragma unroll
    for (int j = 0; j < BF; j++) { 
        const F *__restrict__ const k= __k + ti + tj * j - t;

        s[j].y += www[t + 3] * k[t + 1];
        s[j].z += www[t + 2] * k[t + 1];
        s[j].z += www[t + 3] * k[t + 2];
        s[j].w += www[t + 1] * k[t + 1];
        s[j].w += www[t + 2] * k[t + 2];
        s[j].w += www[t + 3] * k[t + 3];

        F4(x, ti + tj * j) = s[j];
    }
}

template <typename F>
__global__ void kernel_backward(const F *__restrict__ const __w, const F *__restrict__ const __k, const F *__restrict__ const __gwk,
                                F *__restrict__ const gw, F *__restrict__ const gk,
                                const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int t = threadIdx.x << 2;

    __shared__ F w[Tmax];
    __shared__ F k[Tmax];
    __shared__ F gg[Tmax];
    F4(w, t) = F4(__w, t + T * (i % C));
    F4(k, t) = F4(__k, t + T * i);
    F4(gg, t) = F4(__gwk, t + T * i);
    __syncthreads();

    float4 s = {0, 0, 0, 0};
    const F *__restrict__ const ga = gg + T - t - 4;

    for (int u = 0; u <= t; u++) {
        F x = k[u];
        s.x += ga[u + 3] * x;
        s.y += ga[u + 2] * x;
        s.z += ga[u + 1] * x;
        s.w += ga[u] * x;
    }
    s.y += ga[t + 3] * k[t + 1];
    s.z += ga[t + 2] * k[t + 1];
    s.z += ga[t + 3] * k[t + 2];
    s.w += ga[t + 1] * k[t + 1];
    s.w += ga[t + 2] * k[t + 2];
    s.w += ga[t + 3] * k[t + 3];

    F4(gw, t + T * i) = s;

    s.x = 0;
    s.y = 0;
    s.z = 0;
    s.w = 0;
    const F *__restrict__ const gb = gg + T + t - 3;

    for (int u = t + 3; u < T; u++) {
        F x = w[u];
        s.x += gb[2 - u] * x;
        s.y += gb[3 - u] * x;
        s.z += gb[4 - u] * x;
        s.w += gb[5 - u] * x;
    }
    s.x += gb[2 - t] * w[t + 0];
    s.x += gb[1 - t] * w[t + 1];
    s.x += gb[0 - t] * w[t + 2];
    s.y += gb[2 - t] * w[t + 1];
    s.y += gb[1 - t] * w[t + 2];
    s.z += gb[2 - t] * w[t + 2];

    F4(gk, t + T * i) = s;
}

void cuda_forward(const float *w, const float *k, float *x, float eps, int B, int C, int T) {
    dim3 gridDim(1, B * C / BF);
    dim3 blockDim(T >> 2);
    kernel_forward<<<gridDim, blockDim>>>(w, k, x, eps, B, C, T);
}
void cuda_backward(const float *w, const float *k, const float *gwk, float *gw, float *gk, int B, int C, int T) {
    dim3 gridDim(1, B * C);
    dim3 blockDim(T >> 2);
    kernel_backward<<<gridDim, blockDim>>>(w, k, gwk, gw, gk, B, C, T);
}
