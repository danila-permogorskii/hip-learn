/* ============================================================================
 * 00-toolchain-smoke / toolchain_smoke.hip
 * ============================================================================
 *
 * SYFTE — PURPOSE
 *   This is rung zero of the ck_tile track. The KERNEL is intentionally trivial
 *   (a vector square) so that 100% of your attention goes to the DEV LOOP:
 *   build -> run -> debug -> profile -> trace -> disassemble.
 *   Sedan ÅTERANVÄNDS samma skelett som en bandbreddssond.
 *   (Then the same skeleton is REUSED as a bandwidth probe.)
 *
 * VAD DU LÄR DIG — WHAT YOU LEARN
 *   - The modern ROCm 7.2.4 build invocation (amdclang++, not deprecated hipcc).
 *   - Runtime device query on a gfx950 VF (do NOT trust spec sheets — measure).
 *   - The error-checking discipline that makes GPU bugs visible.
 *   - hipEvent timing -> your first MEASURED HBM bandwidth number.
 *     Detta tal är den första datapunkten i din taklinje (roofline).
 *
 * MÅL-HÅRDVARA — TARGET HARDWARE
 *   AMD Instinct MI350X, gfx950, CDNA4. Wavefront = 64 (still!), LDS = 160 KB/CU.
 *   Running as an SR-IOV virtual function inside a DigitalOcean droplet, so the
 *   query may report a PARTITION of the chip — that is why we measure, not assume.
 *
 * BYGG — BUILD (optimised, for profiling)
 *   amdclang++ -std=c++20 -O3 --offload-arch=gfx950 -x hip \
 *       toolchain_smoke.hip -o toolchain_smoke
 *
 * BYGG — BUILD (debug, for rocgdb)
 *   amdclang++ -std=c++20 -O0 -g --offload-arch=gfx950 -x hip \
 *       toolchain_smoke.hip -o toolchain_smoke_dbg
 *
 * KÖR DEV-SLINGAN — RUN THE DEV LOOP (see PART-1 doc, Section 3)
 *   rocgdb ./toolchain_smoke_dbg        # break vector_square_kernel ; run ; info threads
 *   rocprofv3 --stats -- ./toolchain_smoke
 *   rocprofv3 --hip-trace --hsa-trace -- ./toolchain_smoke
 *   amdclang++ ... --save-temps ...     # read *-gfx950.s for the ISA
 *
 * DOKUMENTATION — DOCS THIS UNLOCKS (read AFTER you run it)
 *   HIP basics .... https://rocm.docs.amd.com/projects/HIP/en/latest/
 *   CDNA basics ... .../conceptual/ck_tile/hardware/gpu_basics.html
 *   The "why" ..... .../conceptual/ck_tile/introduction_motivation.html
 *                   (its "GPU Memory Problem" IS the number you measure below)
 * ============================================================================ */

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

/* ----------------------------------------------------------------------------
 * HIP_CHECK — felhantering (error handling)
 * GPU errors are frequently SILENT. Without this macro a bad launch just
 * produces wrong numbers with no warning. Errors go to stderr, not stdout
 * (stdout is for results; stderr is for problems — keep them separable).
 * -------------------------------------------------------------------------- */
#define HIP_CHECK(call)                                                                        \
    do {                                                                                       \
        hipError_t err_ = (call);                                                              \
        if (err_ != hipSuccess) {                                                              \
            std::fprintf(stderr, "HIP error: %s (%s:%d)\n", hipGetErrorString(err_), __FILE__, \
                         __LINE__);                                                            \
            std::exit(EXIT_FAILURE);                                                           \
        }                                                                                      \
    } while (0)

/* ----------------------------------------------------------------------------
 * KÄRNA 1 — KERNEL 1: vector_square_kernel
 * The breakpoint target. Trivial work, so it is easy to inspect lane state.
 *
 *   __global__   : runs on the device (GPU), called from the host (CPU).
 *   __restrict__ : promises the pointers do not alias -> the compiler may
 *                  reorder/vectorise loads & stores. Always use it on kernel
 *                  pointers. (Lova att pekarna inte överlappar.)
 *   uint32_t n   : unsigned -> no negative sizes, and it saves a register vs int.
 *
 * Trådindex — thread index:
 *   global id = blockIdx.x * blockDim.x + threadIdx.x
 *   We launch MORE threads than elements (to fill whole wavefronts of 64), so
 *   the bounds check is NOT optional — extra lanes must touch no memory.
 * -------------------------------------------------------------------------- */
__global__ void vector_square_kernel(float* __restrict__ out, const float* __restrict__ in,
                                     uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * in[tid];  // <-- set a rocgdb breakpoint on this line
    }
}

/* ----------------------------------------------------------------------------
 * KÄRNA 2 — KERNEL 2: copy_kernel  (the REUSE — bandbreddssond / bandwidth probe)
 *
 * Same launch/index skeleton as KERNEL 1, but now the POINT is the memory
 * traffic, not the arithmetic. A grid-stride loop lets a fixed-size grid stream
 * an arbitrarily large buffer: each lane walks the array in steps of the total
 * thread count, so consecutive lanes touch consecutive addresses on each step
 * -> COALESCED access (sammanslagen åtkomst), the friendliest pattern for HBM.
 *
 * Bytes moved per element = 1 read (in) + 1 write (out) = 2 * sizeof(float).
 * That factor of 2 is what turns elapsed-time into GB/s further down.
 * -------------------------------------------------------------------------- */
__global__ void copy_kernel(float* __restrict__ out, const float* __restrict__ in, uint32_t n) {
    uint32_t stride = gridDim.x * blockDim.x;  // total live lanes
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = in[i];
    }
}

/* ----------------------------------------------------------------------------
 * print_device — enhetsförfrågan (device query)
 * Learn YOUR VF's real numbers. On CDNA4 the wavefront is still 64, but the
 * constexpr macro __AMDGCN_WAVEFRONT_SIZE__ is deprecated in ROCm 7.x — query
 * prop.warpSize at runtime instead of hardcoding a literal.
 * -------------------------------------------------------------------------- */
static void print_device(int dev) {
    hipDeviceProp_t p{};
    HIP_CHECK(hipGetDeviceProperties(&p, dev));

    std::printf("=== Enhet / Device %d ===\n", dev);
    std::printf("  Arch (gcnArchName)        : %s\n", p.gcnArchName);
    std::printf("  Wavefront (warpSize)      : %d  (runtime-queried, not hardcoded)\n", p.warpSize);
    std::printf("  Compute Units (CUs)       : %d\n", p.multiProcessorCount);
    std::printf("  LDS per block             : %zu KB\n", p.sharedMemPerBlock / 1024);
    std::printf("  LDS per CU (multiproc)    : %zu KB  (CDNA4 target: 160)\n",
                p.sharedMemPerMultiprocessor / 1024);
    std::printf("  Global memory (this VF)   : %.1f GB\n",
                static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));

    // Theoretical peak HBM bandwidth, IF the VF exposes the clock/bus fields.
    // Teoretisk topp = 2 (DDR) * klockfrekvens * bussbredd. Guarded — may be 0 on a VF.
    if (p.memoryClockRate > 0 && p.memoryBusWidth > 0) {
        double peak_gbs = 2.0 * (p.memoryClockRate * 1e3) * (p.memoryBusWidth / 8.0) / 1e9;
        std::printf("  Theoretical HBM peak      : %.0f GB/s  (cross-check with amd-smi)\n",
                    peak_gbs);
    } else {
        std::printf("  Theoretical HBM peak      : (not exposed by VF — use amd-smi)\n");
    }
    std::printf("\n");
}

int main() {
    int dev = 0;
    HIP_CHECK(hipSetDevice(dev));
    print_device(dev);

    constexpr uint32_t kBlock = 256;  // multiple of 64 -> no partial wavefronts

    /* ====================================================================
     * DEL A — PART A: vector square (felsöknings- & profileringsmål)
     *                 (the debug & profiling target)
     * ==================================================================== */
    {
        constexpr uint32_t N = 1u << 20;  // 1,048,576 elements
        const size_t bytes = static_cast<size_t>(N) * sizeof(float);

        // PINNED host memory (hipHostMalloc) — ~3x faster transfers than malloc,
        // and DMA-friendly. Aldrig malloc för buffertar som rör GPU:n.
        float *h_in, *h_out;
        HIP_CHECK(hipHostMalloc(&h_in, bytes));
        HIP_CHECK(hipHostMalloc(&h_out, bytes));
        for (uint32_t i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 100);

        // Device (HBM) allocations.
        float *d_in, *d_out;
        HIP_CHECK(hipMalloc(&d_in, bytes));
        HIP_CHECK(hipMalloc(&d_out, bytes));

        HIP_CHECK(hipMemcpy(d_in, h_in, bytes, hipMemcpyHostToDevice));

        uint32_t grid = (N + kBlock - 1) / kBlock;  // enough blocks to cover N
        vector_square_kernel<<<grid, kBlock>>>(d_out, d_in, N);

        // Catch LAUNCH errors (bad config) AND execution errors (after sync).
        HIP_CHECK(hipGetLastError());       // launch-time
        HIP_CHECK(hipDeviceSynchronize());  // execution-time

        HIP_CHECK(hipMemcpy(h_out, d_out, bytes, hipMemcpyDeviceToHost));

        // Verify on host — verifiera. element 7 -> 7*7 = 49.
        bool ok = true;
        for (uint32_t i = 0; i < N; ++i) {
            float expect = h_in[i] * h_in[i];
            if (h_out[i] != expect) {
                ok = false;
                break;
            }
        }
        std::printf("[DEL A] vector square: %s\n", ok ? "KORREKT (correct)" : "FEL (wrong)");

        HIP_CHECK(hipFree(d_in));
        HIP_CHECK(hipFree(d_out));
        HIP_CHECK(hipHostFree(h_in));
        HIP_CHECK(hipHostFree(h_out));
    }

    /* ====================================================================
     * DEL B — PART B: HBM bandwidth probe (ÅTERANVÄNDNING / the reuse)
     * Same skeleton, new question: how many GB/s does THIS VF actually give?
     * For a clean number, no other GPU workload should run concurrently —
     * HBM-bandbreddskonkurrens förstör mätningen (bandwidth contention ruins it).
     * ==================================================================== */
    {
        constexpr uint32_t N = 1u << 26;  // 67,108,864 elements (256 MB)
        const size_t bytes = static_cast<size_t>(N) * sizeof(float);

        float *d_in, *d_out;
        HIP_CHECK(hipMalloc(&d_in, bytes));
        HIP_CHECK(hipMalloc(&d_out, bytes));
        HIP_CHECK(hipMemset(d_in, 1, bytes));  // touch the pages

        // A fixed grid; the grid-stride loop handles the rest.
        // 4 blocks/CU is a reasonable starting occupancy — vi mäter, vi gissar inte.
        uint32_t grid = static_cast<uint32_t>(4 * /*CUs*/ 304);

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Warmup — uppvärmning (first launch pays one-off costs; exclude it).
        copy_kernel<<<grid, kBlock>>>(d_out, d_in, N);
        HIP_CHECK(hipDeviceSynchronize());

        constexpr int kIters = 50;
        HIP_CHECK(hipEventRecord(start));
        for (int it = 0; it < kIters; ++it) copy_kernel<<<grid, kBlock>>>(d_out, d_in, N);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        double sec_per_iter = (ms / 1000.0) / kIters;
        double moved = 2.0 * static_cast<double>(bytes);  // read + write
        double gbs = moved / sec_per_iter / 1e9;

        std::printf("[DEL B] copy %.0f MB x%d -> %.1f GB/s uppmätt (measured)\n",
                    static_cast<double>(bytes) / (1024.0 * 1024.0), kIters, gbs);
        std::printf("        Jämför med din VF:s topp via amd-smi.\n");
        std::printf("        (Compare against your VF peak. The %% you hit IS your\n");
        std::printf("         roofline ceiling for memory-bound kernels.)\n");

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_in));
        HIP_CHECK(hipFree(d_out));
    }

    std::printf(
        "\nKlart. (Done.) Nu: kör dev-slingan på denna fil. "
        "(Now: run the dev loop on this file.)\n");
    return 0;
}
