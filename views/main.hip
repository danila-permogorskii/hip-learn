#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <iostream>

#define HIP_CHECK(call) \
    do { \
        hipError_t err_ = (call); \
        if (err_ != hipSuccess) { \
            std::fprintf(stderr, "HIP Error: %s (%s:%d)\n", \
                hipGetErrorString(err_), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0) \

template <typename T>
struct BufferView {
    T* __restrict__ p; // the bytes (global or lds or ...)
    uint32_t n;    // element count - a view knows its own bounds

    __host__ __device__ bool in_range(uint32_t i) const { return i < n; }

    // get / set / update - the three operation families CK's buffer_view exposes
    __host__ __device__ T get (uint32_t i) const { return p[i]; }
    __host__ __device__ void set(uint32_t i, T v) const { p[i] = v; }
    __host__ __device__ void update_add(uint32_t i, T v) const { p[i] += v; }
};

__global__ void square_via_lds_view(float* __restrict__ out,
                                    const float* __restrict__ in,
                                    uint32_t n) {
    __shared__ float tile[256];

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t local = threadIdx.x;

    if (tid < n) {
        tile[local] = in[tid];
    }
    __syncthreads();

    if (tid < n) {
        float x = tile[local];
        out[tid] = x * x;
    }
}

int main() {
    std::cout << "Everything is ok" << '\n';



    return 0;
}