/* ============================================================================
 * 03-coordinate-systems / 03-coordinate-systems.hip
 * ============================================================================
 *
 * SYFTE — PURPOSE  (Path B: Koordinatsystem / Coordinate systems — det branta steget)
 *   A tile is computed by MANY lanes at once, each holding SEVERAL elements in
 *   registers. The new question: vilket körfält äger vilket element, och i vilken
 *   register-plats? (which lane owns which element, and in which register slot?)
 *
 *   Three spaces answer it:
 *     X = WHAT element (logical row,col in the tile)
 *     P = WHO owns it  (the lane / partition)
 *     Y = WHERE in that lane (the register slot)
 *   The distribution is the map  (P, Y) -> X.  It STACKS on Part 3:
 *     (P,Y) --distribution--> X --Embed (descriptor)--> byte offset.
 *
 * BYGG — BUILD  (arch-neutral fat binary)
 *   amdclang++ -std=c++20 -O3 -x hip \
 *       --offload-arch=gfx942 --offload-arch=gfx950 \
 *       03-coordinate-systems.hip -o coords
 *
 * DOCS THIS UNLOCKS — coordinate_systems.html, tensor_coordinates.html
 * ============================================================================ */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err_ = (call);                                               \
        if (err_ != hipSuccess) {                                               \
            std::fprintf(stderr, "HIP error: %s (%s:%d)\n",                      \
                         hipGetErrorString(err_), __FILE__, __LINE__);          \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

/* ----------------------------------------------------------------------------
 * STEG 1 — P only: Y has length 1, so X = P. One element per lane.
 * Just nails the vocabulary: P = who, X = what. (Familiar vector-square.)
 * -------------------------------------------------------------------------- */
__global__ void square_p_only(float* __restrict__ out,
                              const float* __restrict__ in, uint32_t n) {
    uint32_t P = blockIdx.x * blockDim.x + threadIdx.x;   // WHO
    if (P < n) {
        uint32_t X = P;                                   // (P, Y=0) -> X (VGPR)
        out[X] = in[X] * in[X];
    }
}

/* ----------------------------------------------------------------------------
 * STEG 2 — add Y (each lane owns V elements). SAME data, two distributions.
 *   cyclic : X(P,Y) = Y*Pcount + P  -> fixed Y, consecutive lanes -> COALESCED
 *   blocked: X(P,Y) = P*V + Y       -> fixed Y, lanes V apart     -> UNCOALESCED
 * The cyclic map is exactly the grid-stride loop you have always used.
 * -------------------------------------------------------------------------- */
__global__ void copy_cyclic(float* __restrict__ out,
                            const float* __restrict__ in, uint32_t n, uint32_t V) {
    uint32_t P      = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t Pcount = gridDim.x * blockDim.x;
    for (uint32_t Y = 0; Y < V; ++Y) {
        uint32_t X = Y * Pcount + P;                      // (P,Y) -> X  [coalesced]
        if (X < n) out[X] = in[X];
    }
}
__global__ void copy_blocked(float* __restrict__ out,
                             const float* __restrict__ in, uint32_t n, uint32_t V) {
    uint32_t P = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t Y = 0; Y < V; ++Y) {
        uint32_t X = P * V + Y;                           // (P,Y) -> X  [uncoalesced]
        if (X < n) out[X] = in[X];
    }
}

/* ----------------------------------------------------------------------------
 * STEG 3 — a 2-D fragment: (P, Y) -> (row, col). 16x16 tile, 64 lanes x 4 slots.
 * The SHAPE of an MFMA fragment. Each lane gathers its 4 owned elements.
 * NB: this is a representative distribution (flatX = Y*64 + P), NOT the exact
 * v_mfma_f64_16x16x4 register layout — that is derived from the ISA in Part 8.
 * Formen på frågan, inte det exakta svaret (the shape of the question).
 * -------------------------------------------------------------------------- */
__global__ void gather_fragment(float* __restrict__ frag_out,
                                const float* __restrict__ in /* 256 = 16x16 */) {
    uint32_t P = threadIdx.x;                              // lane 0..63 (WHO)
    for (uint32_t Y = 0; Y < 4; ++Y) {                    // 4 register slots (WHERE)
        uint32_t flatX = Y * 64 + P;                      // (P,Y) -> flat logical X
        uint32_t row = flatX / 16, col = flatX % 16;      // -> 2-D tile coordinate (WHAT)
        frag_out[P * 4 + Y] = in[row * 16 + col];         // gather into this lane's slot
    }
}

template <typename Launch>
static double measure_gbs(Launch launch, size_t bytes, int kIters) {
    hipEvent_t s, e;
    HIP_CHECK(hipEventCreate(&s)); HIP_CHECK(hipEventCreate(&e));
    launch(); HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(s));
    for (int it = 0; it < kIters; ++it) launch();
    HIP_CHECK(hipEventRecord(e)); HIP_CHECK(hipEventSynchronize(e));
    float ms = 0.f; HIP_CHECK(hipEventElapsedTime(&ms, s, e));
    HIP_CHECK(hipEventDestroy(s)); HIP_CHECK(hipEventDestroy(e));
    return (2.0 * double(bytes)) / ((ms / 1000.0) / kIters) / 1e9;
}

int main() {
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{}; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("running on: %s\n\n", prop.gcnArchName);
    constexpr uint32_t kBlock = 256;

    /* ---- STEG 1: P only, X = P ---- */
    {
        constexpr uint32_t N = 1u << 20;
        const size_t bytes = size_t(N) * sizeof(float);
        std::vector<float> h(N), out(N);
        for (uint32_t i = 0; i < N; ++i) h[i] = float(i % 100);

        float *din, *dout;
        HIP_CHECK(hipMalloc(&din, bytes)); HIP_CHECK(hipMalloc(&dout, bytes));
        HIP_CHECK(hipMemcpy(din, h.data(), bytes, hipMemcpyHostToDevice));
        square_p_only<<<(N + kBlock - 1) / kBlock, kBlock>>>(dout, din, N); // 4096 blocks, 256 threads per block
        HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(out.data(), dout, bytes, hipMemcpyDeviceToHost));

        bool ok = true;
        for (uint32_t i = 0; i < N; ++i) if (out[i] != h[i]*h[i]) { ok = false; break; }
        std::printf("[STEG 1] P-only (X = P): %s\n", ok ? "KORREKT" : "FEL (wrong)");
        HIP_CHECK(hipFree(din)); HIP_CHECK(hipFree(dout));
    }

    /* ---- STEG 2: cyclic vs blocked over the SAME tile, measured ---- */
    {
        constexpr uint32_t V = 64;
        uint32_t grid   = uint32_t(4 * prop.multiProcessorCount);
        uint32_t Pcount = grid * kBlock;
        uint32_t N      = Pcount * V;                     // tile covers exactly Pcount*V
        const size_t bytes = size_t(N) * sizeof(float);

        std::vector<float> h(N), oc(N), ob(N);
        for (uint32_t i = 0; i < N; ++i) h[i] = float(i % 1000);
        float *din, *doc, *dob;
        HIP_CHECK(hipMalloc(&din, bytes));
        HIP_CHECK(hipMalloc(&doc, bytes)); HIP_CHECK(hipMalloc(&dob, bytes));
        HIP_CHECK(hipMemcpy(din, h.data(), bytes, hipMemcpyHostToDevice));

        copy_cyclic <<<grid, kBlock>>>(doc, din, N, V);
        copy_blocked<<<grid, kBlock>>>(dob, din, N, V);
        HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(oc.data(), doc, bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(ob.data(), dob, bytes, hipMemcpyDeviceToHost));

        bool ok = true;                                   // both are full permutations -> identity
        for (uint32_t i = 0; i < N; ++i) if (oc[i] != h[i] || ob[i] != h[i]) { ok = false; break; }
        std::printf("[STEG 2] cyclic & blocked both correct copies: %s\n",
                    ok ? "KORREKT" : "FEL (wrong)");

        double gc = measure_gbs([&]{ copy_cyclic <<<grid,kBlock>>>(doc,din,N,V); }, bytes, 50);
        double gb = measure_gbs([&]{ copy_blocked<<<grid,kBlock>>>(dob,din,N,V); }, bytes, 50);
        std::printf("         cyclic  (coalesced): %.1f GB/s\n", gc);
        std::printf("         blocked (strided)  : %.1f GB/s\n", gb);
        std::printf("         -> samma data, fördelningen avgör farten (%.2fx).\n", gc / gb);
        std::printf("            (same data; the (P,Y)->X distribution decides the speed.)\n");
        HIP_CHECK(hipFree(din)); HIP_CHECK(hipFree(doc)); HIP_CHECK(hipFree(dob));
    }

    /* ---- STEG 3: 16x16 MFMA-shaped fragment, (P,Y) -> (row,col) ---- */
    {
        constexpr uint32_t T = 16 * 16;                   // 256 tile elements
        const size_t tbytes = T * sizeof(float);
        std::vector<float> h(T), frag(T);
        for (uint32_t i = 0; i < T; ++i) h[i] = float(i);

        float *din, *dfrag;
        HIP_CHECK(hipMalloc(&din, tbytes)); HIP_CHECK(hipMalloc(&dfrag, tbytes));
        HIP_CHECK(hipMemcpy(din, h.data(), tbytes, hipMemcpyHostToDevice));
        gather_fragment<<<1, 64>>>(dfrag, din);           // 64 lanes, 4 slots each
        HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(frag.data(), dfrag, tbytes, hipMemcpyDeviceToHost));

        bool ok = true;                                   // frag[P*4+Y] must == h[Y*64+P]
        for (uint32_t P = 0; P < 64 && ok; ++P)
            for (uint32_t Y = 0; Y < 4; ++Y)
                if (frag[P * 4 + Y] != h[Y * 64 + P]) { ok = false; break; }
        std::printf("[STEG 3] fragment gather (P,Y)->(row,col): %s\n",
                    ok ? "KORREKT" : "FEL (wrong)");
        HIP_CHECK(hipFree(din)); HIP_CHECK(hipFree(dfrag));
    }

    std::printf("\nKlart. Läs nu coordinate_systems.html.\n");
    return 0;
}
