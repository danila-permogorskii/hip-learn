#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#define HIP_CHECK(call)                                                                        \
    do {                                                                                       \
        hipError_t err_ = (call);                                                              \
        if (err_ != hipSuccess) {                                                              \
            std::fprintf(stderr, "HIP Error: %s (%s:%d)\n", hipGetErrorString(err_), __FILE__, \
                         __LINE__);                                                            \
            std::exit(EXIT_FAILURE);                                                           \
        }                                                                                      \
    } while (0)

template <typename T>
struct BufferView {
    T* __restrict__ p;  // the bytes (global or lds or ...)
    uint32_t n;         // element count - a view knows its own bounds

    __host__ __device__ bool in_range(uint32_t i) const {
        return i < n;
    }

    // get / set / update - the three operation families CK's buffer_view exposes
    __host__ __device__ T get(uint32_t i) const {
        return p[i];
    }
    __host__ __device__ void set(uint32_t i, T v) const {
        p[i] = v;
    }
    __host__ __device__ void update_add(uint32_t i, T v) const {
        p[i] += v;
    }
};

__global__ void square_via_view(float* __restrict__ out, const float* __restrict__ in, uint32_t n) {
    BufferView<const float> vin{in, n};
    BufferView<float> vout{out, n};
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (vin.in_range(tid)) {
        float x = vin.get(tid);
        vout.set(tid, x * x);
    }
}

__global__ void square_via_lds_view(float* __restrict__ out, const float* __restrict__ in,
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

template <typename T>
struct TensorView2D {
    T* __restrict__ p;
    uint32_t rows, cols;
    uint32_t stride_row, stride_col;

    __host__ __device__ uint32_t offset(uint32_t r, uint32_t c) const {
        return r * stride_row + c * stride_col;
    }

    __host__ __device__ T& at(uint32_t r, uint32_t c) const {
        return p[offset(r, c)];
    }

    __host__ __device__ TensorView2D transposed() const {
        return TensorView2D{p, cols, rows, stride_col, stride_row};
    }
};

__global__ void read_through_view(float* __restrict__ out, const float* __restrict__ in,
                                  uint32_t rows, uint32_t cols, uint32_t sr, uint32_t sc) {
    TensorView2D<const float> v{in, rows, cols, sr, sc};
    uint32_t r = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        out[r * cols + c] = v.at(r, c);  // out is plain row-major
    }
}

__global__ void copy_coalesced(float* __restrict__ out, const float* __restrict__ in, uint32_t n) {
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) out[i] = in[i];
}

__global__ void copy_strided(float* __restrict__ out, const float* __restrict__ in, uint32_t n,
                             uint32_t side) {
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint32_t r = i / side, c = i % side;
        uint32_t src = c * side + r;
        out[i] = in[src];
    }
}

template <typename Launch>
static double measure_gbs(Launch launch, size_t bytes, int kIters) {
    hipEvent_t s, e;
    HIP_CHECK(hipEventCreate(&s));
    HIP_CHECK(hipEventCreate(&e));
    launch();
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(s));
    for (int it = 0; it < kIters; ++it) launch();
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));
    HIP_CHECK(hipEventDestroy(s));
    HIP_CHECK(hipEventDestroy(e));
    double sec = (ms / 1000.0) / kIters;
    return (2.0 * static_cast<double>(bytes)) / sec / 1e9;
}

int main() {
    std::cout << "Everything is ok" << '\n';

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("runninng on: %s \n\n", prop.gcnArchName);

    constexpr uint32_t kBlock = 256;

    // Buffer view and the lds-view variations
    {
        constexpr uint32_t N = 1u << 20;
        const size_t bytes = size_t(N) * sizeof(float);
        std::vector<float> h(N), r1(N), r2(N);
        for (uint32_t i = 0; i < N; i++) h[i] = float(i % 100);

        float *din, *dout;
        HIP_CHECK(hipMalloc(&din, bytes));
        HIP_CHECK(hipMalloc(&dout, bytes));
        uint32_t grid = (N + kBlock - 1) / kBlock;

        square_via_view<<<grid, kBlock>>>(dout, din, N);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(r1.data(), dout, bytes, hipMemcpyDeviceToHost));

        square_via_lds_view<<<grid, kBlock>>>(dout, din, N);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(r2.data(), dout, bytes, hipMemcpyDeviceToHost));

        bool ok = true;
        // TODO: Find out why the result is "wrong" at the first cycle step
        for (uint32_t i = 0; i < N; i++) {
            if (r1[i] != h[i] * h[i] || r2[i] != h[i] * h[i]) {
                ok = false;
                break;
            }
        }

        std::printf("Buffer View (global) + LDS-view: %s\n", ok ? "correct" : "wrong");

        HIP_CHECK(hipFree(din));
        HIP_CHECK(hipFree(dout));
    }

    {
        constexpr uint32_t R = 512, C = 512, N = R * C;
        const size_t bytes = size_t(N) * sizeof(float);
        std::vector<float> h(N), out(N);
        for (uint32_t i = 0; i < N; i++) {
            h[i] = float(i);  // row-major fill
        }

        float *din, *dout;
        HIP_CHECK(hipMalloc(&din, bytes));
        HIP_CHECK(hipMalloc(&dout, bytes));

        dim3 block(16, 16), grid((C + 15) / 16, (R + 15) / 16);
        // Transpose-READ: feed the input view its column-major strides (swapped).
        // Source is row-major (sr=C, sc=1); swapping gives sr=1, sc=C -> transpose.
        read_through_view<<<grid, block>>>(dout, din, R, C, 1u, C);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(out.data(), dout, bytes, hipMemcpyDeviceToHost));

        bool ok = true;
        for (uint32_t r = 0; r < R; ++r) {
            for (uint32_t c = 0; c < C; ++c) {
                if (out[r * C + c] != h[c * C + r]) {
                    ok = false;
                    break;
                }
            }
        }

        // TODO: the same problem as in previous block
        std::printf("Transpose by view (no data moved): %s\n", ok ? "correct" : "wrong");
        HIP_CHECK(hipFree(din));
        HIP_CHECK(hipFree(dout));
    }

    {
        constexpr uint32_t side = 4096, N = side * side;
        const size_t bytes = size_t(N) * sizeof(float);
        float *din, *dout;
        HIP_CHECK(hipMalloc(&din, bytes));
        HIP_CHECK(hipMalloc(&dout, bytes));
        HIP_CHECK(hipMemset(din, 1, bytes));
        uint32_t grid = uint32_t(4 * prop.multiProcessorCount);

        double gc = measure_gbs([&] { copy_coalesced<<<grid, kBlock>>>(dout, din, N); }, bytes, 50);
        double gs =
            measure_gbs([&] { copy_strided<<<grid, kBlock>>>(dout, din, N, side); }, bytes, 50);

        std::printf("coalesced: %.1f GB/s\n", gc);
        std::printf("strided: %.1f GB/s\n", gs);
        std::printf("penalty: %.2fx slower for the same bytes\n", gc / gs);

        HIP_CHECK(hipFree(din));
        HIP_CHECK(hipFree(dout));
    }

    return 0;
}
