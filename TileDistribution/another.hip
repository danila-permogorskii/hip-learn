#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = (call); \
        if (err != hipSuccess) { \
            std::fprintf(stderr, "HIP error: %s (%s:%d)\n", \
                hipGetErrorString(err), __FILE__, __LINE__); \
           std::exit(EXIT_FAILURE); \
        } \
    } while (0) \

struct Dist1D {
    uint32_t strideP, strideY, lenY;
    __host__ __device__ uint32_t x(uint32_t P, uint32_t Y) const {
        return P * strideP + Y * strideY;
    }
};

struct FragDist {
    uint32_t pLo; // lanes per row-group
    uint32_t rowsPerLane;

    __host__ __device__ void rc(uint32_t P, uint32_t Y,
        uint32_t& row, uint32_t& col) const {
        uint32_t pHi = P / pLo; // Merge: P -> (pHi, pLo)
        uint32_t pLoIdx = P % pLo;

        col = pLoIdx;
        row = pHi * rowsPerLane + Y;
    }
};

// Tensor descriptor: (row, col) -> offset
struct Embed {
    uint32_t sr, sc;
    __host__ __device__ uint32_t off(uint32_t r, uint32_t c) const {
        return r * sr + c * sc;
    }
};

__global__ void copy_dist(float* __restrict__ out, const float* __restrict__ in,
    Dist1D d, uint32_t n) {
        uint32_t P = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint32_t Y = 0; Y < d.lenY; Y++) {
            uint32_t X = d.x(P, Y);
            if (X < n) out[X] = in[X];
        }
    }

__global__ void gather_frag(float* __restrict__ frag, const float* __restrict__ in,
    FragDist d) {
    uint32_t P = threadIdx.x;
    for (uint32_t Y = 0; Y < d.rowsPerLane; Y++) {
        uint32_t row, col;
        d.rc(P, Y, row, col);
        frag[P * d.rowsPerLane + Y] = in[row * 16 + col];
    }
}

__global__ void load_tile(float* __restrict__ frag, const float* __restrict__ in,
    FragDist d, Embed desc) {
        uint32_t P = threadIdx.x;
        for (uint32_t Y = 0; Y < d.rowsPerLane; Y++) {
            uint32_t row, col;
            d.rc(P, Y, row, col);
            frag[P * d.rowsPerLane + Y] = in[desc.off(row, col)];
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
    for (int it = 0; it < kIters; it++) {
        launch();
    }
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));

    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));
    HIP_CHECK(hipEventDestroy(s));
    HIP_CHECK(hipEventDestroy(e));
    return (2.0 * double(bytes)) / ((ms / 1000.0) / kIters) / 1e9;
}

int main() {
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop,0));
    std::printf("running on: %s\n\n", prop.gcnArchName);
    constexpr uint32_t kBlock = 256;

    {
        constexpr uint32_t V = 64;
        uint32_t grid = uint32_t(4 * prop.multiProcessorCount);
        uint32_t Pcount = grid * kBlock;
        uint32_t N = Pcount * V;
        const size_t bytes = size_t(N) * sizeof(float);

        std::vector<float> h(N), oc(N), ob(N);
        for (uint32_t i = 0; i < N; i++) {
            h[i] = float(i % 1000);
        }
        float *din, *doc, *dob;
        HIP_CHECK(hipMalloc(&din, bytes));
        HIP_CHECK(hipMalloc(&doc, bytes));
        HIP_CHECK(hipMalloc(&dob, bytes));

        HIP_CHECK(hipMemcpy(din, h.data(), bytes, hipMemcpyHostToDevice));

        Dist1D cyclic {1u, Pcount, V};
        Dist1D blocked {V, 1u, V};

        copy_dist<<<grid, kBlock>>>(doc, din, cyclic, N);
        copy_dist<<<grid, kBlock>>>(dob, din, blocked, N);

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(oc.data(), doc, bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(ob.data(), dob, bytes, hipMemcpyDeviceToHost));

        bool ok = true;
        for (uint32_t i = 0; i < N; ++i) if (oc[i] != h[i] || ob[i] != h[i]) { ok = false; break; }
        std::printf("[STEG 1] same engine, cyclic & blocked as data: %s\n",
                    ok ? "KORREKT" : "FEL (wrong)");

        double gc = measure_gbs([&]{ copy_dist<<<grid,kBlock>>>(doc,din,cyclic, N); }, bytes, 50);
        double gb = measure_gbs([&]{ copy_dist<<<grid,kBlock>>>(dob,din,blocked,N); }, bytes, 50);
        std::printf("         cyclic %.1f GB/s | blocked %.1f GB/s | %.2fx\n", gc, gb, gc/gb);
        HIP_CHECK(hipFree(din)); HIP_CHECK(hipFree(doc)); HIP_CHECK(hipFree(dob));
    }

    {
        constexpr uint32_t T = 16 * 16;
        const size_t tb = T * sizeof(float);
        std::vector<float> h(T), frag(T);
        for (uint32_t i = 0; i < T; i++)
            h[i] = float(i);

        float *din, *dfrag;
        HIP_CHECK(hipMalloc(&din, tb));
        HIP_CHECK(hipMalloc(&dfrag, tb));
        HIP_CHECK(hipMemcpy(din, h.data(), tb, hipMemcpyHostToDevice));

        FragDist fd {16u, 4u};
        gather_frag<<<1, 64>>>(dfrag, din, fd);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(frag.data(), dfrag, tb, hipMemcpyDeviceToHost));

        bool ok = true;
        for (uint32_t P = 0; P < 64; P++) {
            for (uint32_t Y = 0; Y < 4; Y++) {
                uint32_t pHi = P / 16, col = P % 16, row = pHi * 4 + Y;
                if (frag[P*4 + Y] != h[row*16 + col]) {
                    ok = false;
                    break;
                }
            }
        }

        std::printf("[STEG 2] declared 2-D fragment gather: %s\n", ok ? "KORREKT" : "FEL (wrong)");
          HIP_CHECK(hipFree(din)); HIP_CHECK(hipFree(dfrag));

    }
}
