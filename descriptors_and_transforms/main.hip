#include <cstdio>
#include <hip/hip_runtime.h>

#define HIP_CHECK(call) \
    do { \
        hipError_t _err = (call); \
        if (_err != hipSuccess) { \
            std::fprintf(stderr, "HIP Error: %s (%s:%d)\n", hipGetErrorString(_err), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

/* ============================================================================
 *  INDEX TYPES
 * ============================================================================ */

struct Idx2 {
    uint32_t r, c; // a 2D visible index — a point on a logical grid
};

/* ============================================================================
 *  EMBED  —  visible (r,c) → physical 1D offset
 *
 *  Formula:  offset = r * sr + c * sc
 *
 *  Think of it as "folding" a 2D grid into a flat memory array.
 *  The strides (sr, sc) control HOW you fold:
 *
 *    Embed{sr:4, sc:1}  →  row-major, compact:
 *      (0,0)→0  (0,1)→1  (0,2)→2  (0,3)→3
 *      (1,0)→4  (1,1)→5  (1,2)→6  (1,3)→7
 *
 *    Embed{sr:6, sc:1}  →  row-major, padded (lda=6, 2 wasted per row):
 *      (0,0)→0  (0,1)→1  (0,2)→2  (0,3)→3   [4,5 = padding]
 *      (1,0)→6  (1,1)→7  (1,2)→8  (1,3)→9   [10,11 = padding]
 *
 *    Embed{sr:1, sc:3}  →  column-major (transpose):
 *      (0,0)→0  (0,1)→3  (0,2)→6
 *      (1,0)→1  (1,1)→4  (1,2)→7
 *      (2,0)→2  (2,1)→5  (2,2)→8
 *
 *  Transpose is just Embed with sr/sc swapped. No index swapping needed.
 * ============================================================================ */
struct Embed {
    uint32_t sr, sc;
    __host__ __device__ uint32_t lower(Idx2 u) const {
        return u.r * sr + u.c * sc;
    }
};

/* ============================================================================
 *  MERGE  —  flat 1D index → visible (r,c)
 *
 *  Formula:  row = flat / cols,  col = flat % cols
 *
 *  Despite the name "merge", it SPLITs one dimension into two.
 *  It unrolls a linear index into row-major 2D coordinates.
 *
 *  Example:  Merge{cols:4}
 *
 *    flat=0  →  0/4=0, 0%4=0  →  (0,0)
 *    flat=1  →  1/4=0, 1%4=1  →  (0,1)
 *    flat=3  →  3/4=0, 3%4=3  →  (0,3)
 *    flat=4  →  4/4=1, 4%4=0  →  (1,0)   ← row wraps
 *    flat=7  →  7/4=1, 7%4=3  →  (1,3)
 *
 *  Visual:
 *        ┌───┬───┬───┬───┐
 *        │ 0 │ 1 │ 2 │ 3 │   ← flat indices, row 0
 *        ├───┼───┼───┼───┤
 *        │ 4 │ 5 │ 6 │ 7 │   ← flat indices, row 1
 *        └───┴───┴───┴───┘
 * ============================================================================ */
struct Merge {
    uint32_t cols;
    __host__ __device__ Idx2 lower(uint32_t flat) const {
        return {flat / cols, flat % cols};
    }
};

/* ============================================================================
 *  DESCRIPTOR2D  —  composition: flat → Merge → (r,c) → Embed → physical
 *
 *  Pipeline:
 *
 *    flat index
 *        │
 *        ▼
 *    ┌─────────┐
 *    │  Merge  │   "unravel 1D → 2D grid coordinates"
 *    └─────────┘
 *        │
 *        ▼
 *    ┌─────────┐
 *    │  Embed  │   "fold 2D → 1D memory offset (stride/padding choice)"
 *    └─────────┘
 *        │
 *        ▼
 *    physical offset
 *
 *  Padding enters as a STRIDE choice: sr = lda (>= cols) widens the
 *  physical row while the visible (r,c) space is unchanged.
 *
 *  Example:  Descriptor2D { Merge{cols:4}, Embed{sr:6, sc:1} }
 *
 *    flat  →  Merge  →  (r,c)  →  Embed(r×6+c×1)  →  physical
 *      0  →   (0,0)  →   0×6+0×1 = 0
 *      1  →   (0,1)  →   0×6+1×1 = 1
 *      3  →   (0,3)  →   0×6+3×1 = 3
 *      4  →   (1,0)  →   1×6+0×1 = 6    ← jumped over 4,5 (padding)
 *      7  →   (1,3)  →   1×6+3×1 = 9    ← jumped over 10,11 (padding)
 *
 *    Memory layout of destination:
 *      [ data │ data │ data │ data │ PAD │ PAD │ data │ data │ data │ data │ PAD │ PAD ]
 *        0      1      2      3      4     5      6      7      8      9     10    11
 * ============================================================================ */
struct Descriptor2D {
    Merge merge; // flat -> (r,c)
    Embed embed; // (r,c) -> physical offset
    __host__ __device__ uint32_t offset(uint32_t flat) const {
        return embed.lower(merge.lower(flat));
    }
};

/* ============================================================================
 *  KERNEL: transpose_via_embed
 *
 *  Each thread maps to one output element at (r,c).
 *
 *  Output: written in row-major order  →  out[r * cols + c]
 *  Input:  read via Embed with swapped strides → in[e.lower({r,c})]
 *
 *  When Embed{sr:1, sc:C}:
 *    in[r×1 + c×C] reads the input in COLUMN-MAJOR order.
 *    out[r×C + c]  writes the output in ROW-MAJOR order.
 *    Column-major read + row-major write = transpose.
 *
 *  Example: 512×512 matrix, Embed{sr:1, sc:512}
 *
 *    Thread at (r:3, c:7):
 *      out[3×512 + 7]   = out[1543]     ← row-major output position
 *      in[3×1 + 7×512]  = in[3587]      ← column-major input position
 *
 *    Input element at (7,3) = in[7×512+3] = in[3587]
 *    Output element at (3,7) = out[3×512+7] = out[1543]
 *    So: out[3,7] = in[7,3]  →  transpose ✓
 *
 *  Grid layout: 32×32 blocks, each 16×16 threads.
 *    Total: 32×32 × 16×16 = 262144 threads = 512×512 elements.
 *    One thread per output element.
 * ============================================================================ */
__global__ void transpose_via_embed(float* __restrict__ out,
    const float* __restrict__ in,
    uint32_t rows, uint32_t cols, Embed e) {

    uint32_t r = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        out[r * cols + c] = in[e.lower({r,c})];
    }
}

/* ============================================================================
 *  KERNEL: copy_via_merge
 *
 *  1D grid-stride loop. Each thread processes flat indices i, i+stride, ...
 *
 *  Pipeline per element:
 *    i  →  Merge → (r,c)  →  Embed → physical offset in `in`
 *    out[i] = in[ embed.lower( merge.lower(i) ) ]
 *
 *  Direction: reads from padded source, writes to compact destination.
 *  Effect: STRIPS padding.
 *
 *  Example: Merge{cols:4}, Embed{sr:6, sc:1}, n=8
 *
 *    Source (padded, lda=6):
 *      [ a │ b │ c │ d │ _ │ _ │ e │ f │ g │ h │ _ │ _ ]
 *        0   1   2   3   4   5   6   7   8   9   10  11
 *
 *    Per-element:
 *      i=0 → Merge→(0,0) → Embed→0   → out[0] = in[0]  → a
 *      i=1 → Merge→(0,1) → Embed→1   → out[1] = in[1]  → b
 *      i=4 → Merge→(1,0) → Embed→6   → out[4] = in[6]  → e   ← skipped padding
 *      i=7 → Merge→(1,3) → Embed→9   → out[7] = in[9]  → h
 *
 *    Destination (compact):
 *      [ a │ b │ c │ d │ e │ f │ g │ h ]
 *        0   1   2   3   4   5   6   7
 * ============================================================================ */
__global__ void copy_via_merge(float* __restrict__ out,
    const float* __restrict__ in,
    Merge m, Embed e, uint32_t n) {

    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        Idx2 rc = m.lower(i);   // Merge: shows 1 dim, computes 2
        out[i] = in[e.lower(rc)];   // Embed: shows 2, computes 1
    }
}

/* ============================================================================
 *  KERNEL: copy_through_desc
 *
 *  1D grid-stride loop. Each thread processes flat indices i, i+stride, ...
 *
 *  Pipeline per element:
 *    i  →  Descriptor2D → physical offset in `out`
 *    out[ desc.offset(i) ] = in[i]
 *
 *  Direction: reads from compact source, writes to padded destination.
 *  Effect: INSERTS padding. (Opposite of copy_via_merge.)
 *
 *  Example: Descriptor2D { Merge{cols:4}, Embed{sr:6, sc:1} }, n=8
 *
 *    Source (compact, lda=4):
 *      [ a │ b │ c │ d │ e │ f │ g │ h ]
 *        0   1   2   3   4   5   6   7
 *
 *    Per-element:
 *      i=0 → desc→0   → out[0]  = in[0]  → a
 *      i=1 → desc→1   → out[1]  = in[1]  → b
 *      i=4 → desc→6   → out[6]  = in[4]  → e   ← writes past padding gap
 *      i=7 → desc→9   → out[9]  = in[7]  → h
 *
 *    Destination (padded, lda=6):
 *      [ a │ b │ c │ d │ _ │ _ │ e │ f │ g │ h │ _ │ _ ]
 *        0   1   2   3   4   5   6   7   8   9   10  11
 *
 *  KEY DIFFERENCE from copy_via_merge:
 *    copy_via_merge:  out[i] = in[transform(i)]   → read padded, write compact
 *    copy_through:    out[transform(i)] = in[i]   → read compact, write padded
 * ============================================================================ */
__global__ void copy_through_desc(float* __restrict__ out,
    const float* __restrict__ in,
    Descriptor2D d, uint32_t n_visible) {

    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n_visible; i += stride) {
        out[d.offset(i)] = in[i]; // visible i -> physical (padded)
    }
}

/* ============================================================================
 *  BENCHMARK: measure_gbs
 *
 *  Measures throughput in GB/s for a kernel launch.
 *
 *  Timeline:
 *
 *    ──┬──── warmup ────┬──────────────── timed region ────────────────┬──
 *       │ launch()       │  ┌─iter 0─┐ ┌─iter 1─┐ ... ┌─iter kIters-1─┐ │
 *       │ sync()         │  │launch()│ │launch()│     │launch()        │ │
 *       │                │  └────────┘ └────────┘     └────────────────┘ │
 *       ▼                  ▲                                    ▲       ▼
 *                         event s                                event e
 *                         (start)                                (stop)
 *
 *  Formula:
 *    total_ms     = elapsed(s, e)
 *    per_iter_ms  = total_ms / kIters
 *    total_bytes  = 2.0 × useful_bytes    (×2 for read + write)
 *    GB/s         = total_bytes / (per_iter_ms / 1000) / 1e9
 * ============================================================================ */
template <typename Launch>
static double measure_gbs(Launch launch, size_t useful_bytes, int kIters) {
    hipEvent_t s, e;
    HIP_CHECK(hipEventCreate(&s)); HIP_CHECK(hipEventCreate(&e));

    launch(); HIP_CHECK(hipDeviceSynchronize()); // warmup
    HIP_CHECK(hipEventRecord(s));

    for (int it = 0; it < kIters; it++) {
        launch();
    }

    HIP_CHECK(hipEventRecord(e)); HIP_CHECK(hipEventSynchronize(e));

    float ms = 0.f; HIP_CHECK(hipEventElapsedTime(&ms, s, e));

    return (2.0 * double(useful_bytes)) / ((ms / 1000.0) / kIters) / 1e9;
}

/* ============================================================================
 *  MAIN — Transpose Example Walkthrough
 *
 *  Goal: transpose a 512×512 matrix using Embed{sr:1, sc:512}
 *
 *  Data flow:
 *
 *    Host: h[] = {0, 1, 2, ..., 262143}   (row-major 512×512)
 *         │
 *         │ hipMemcpy H2D
 *         ▼
 *    GPU:  din[]  ── transpose_via_embed ──→  dout[]
 *                              │
 *                              │ Embed{sr:1, sc:512}
 *                              │ Each thread (r,c):
 *                              │   out[r×512+c] = in[r×1+c×512]
 *         │
 *         │ hipMemcpy D2H
 *         ▼
 *    Host: out[]  (verify: out[r,c] == h[c,r])
 *
 *  Grid: 32×32 blocks of 16×16 threads = 262144 threads = R×C
 *
 *  Why Embed{1, C} transposes:
 *
 *    Normal row-major:   offset = r×C + c    "step C down, step 1 right"
 *    Embed{1,C}:         offset = r×1 + c×C  "step 1 down, step C right"
 *
 *    Strides swapped → thread grid walks input in column-major order.
 *    Column-major read + row-major write = transpose.
 *    No explicit index swapping. Just different strides.
 *
 *  Verification:
 *    out[r×C+c] should equal h[c×C+r]
 *    Example: out[3,7] = out[1543]  vs  h[7,3] = h[3587] = 3587.0f
 *    If out[1543] == 3587.0f → correct ✓
 * ============================================================================ */
int main() {
    std::printf("Everything is ok\n");

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{}; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("running on: %s\n\n", prop.gcnArchName);
    constexpr uint32_t kBlock = 256;

    // Embed = named stride map; transpose = swap strides
    {
        constexpr uint32_t R = 512, C = 512, N = R * C;
        const size_t bytes = size_t(N) * sizeof(float);
        std::vector<float> h(N), out(N);

        for (uint32_t i = 0; i < N; i++) {
            h[i] = float(i);
        }


        float *din, *dout;
        HIP_CHECK(hipMalloc(&din, bytes));
        HIP_CHECK(hipMalloc(&dout, bytes));
        HIP_CHECK(hipMemcpy(din, h.data(), bytes, hipMemcpyHostToDevice));
        dim3 block(16, 16), grid((C + 15) / 16, (R + 15) / 16);

        transpose_via_embed<<<grid, block>>>(dout, din, R, C, Embed{1u, C}); // sr - 1u, sc - C
        HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(out.data(), dout, bytes, hipMemcpyDeviceToHost));

        bool ok = true;
        for (uint32_t r = 0; r < R && ok; r++)
            for (uint32_t c = 0; c < C; c++)
                if (out[r*C + c] != h[c*C + r]) {
                    ok = false;
                    break;
                }

        std::printf("Embed transpose (swap strides): %s\n", ok ? "correct" : "wrong");
        HIP_CHECK(hipFree(din)); HIP_CHECK(hipFree(dout));
    }

    // Merge then embed
    {
        constexpr uint32_t R = 1024, C = 1024, N = R * C;
        const size_t bytes = size_t(N) * sizeof(float);
        std::vector<float> h(N), out(N);

        float *din, *dout;

        HIP_CHECK(hipMalloc(&din, bytes)); HIP_CHECK(hipMalloc(&dout,bytes));
        HIP_CHECK(hipMemcpy(din, h.data(), bytes, hipMemcpyHostToDevice));
        uint32_t grid = uint32_t(4 * prop.multiProcessorCount);

        copy_via_merge<<<grid, kBlock>>>(dout, din, Merge{ C }, Embed{ C, 1u}, N);
        HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(out.data(), dout, bytes, hipMemcpyDeviceToHost));

        bool ok = true;
        for (uint32_t i = 0; i < N; i++) {
            if (out[i] != h[i]) {
                ok = false;
                break;
            }
        }

        std::printf("Merge -> Embed 1d over 2d: %s\n", ok ? "correct" : "wrong");
        HIP_CHECK(hipFree(din)); HIP_CHECK(hipFree(dout));
    }

    // Pad + composition -> descriptor
    {
        constexpr uint32_t R = 4000, C = 4000, Nvis = R * C; // C isn't a multiple of 64
        const uint32_t lda = ((C + 63) / 64) * 64; // pad row to 64
        const size_t in_bytes = size_t(Nvis) * sizeof(float);
        const size_t pad_bytes = size_t(R) * lda * sizeof(float);

        std::vector<float> h(Nvis), outp(size_t(R) * lda);

        for (uint32_t i = 0; i < Nvis; i++) {
            h[i] = float(i % 100000);
        }

        float *din, *dpack, *dpad;
        HIP_CHECK(hipMalloc(&din, in_bytes));
        HIP_CHECK(hipMalloc(&dpack, in_bytes));
        HIP_CHECK(hipMalloc(&dpad, pad_bytes));
        HIP_CHECK(hipMemcpy(din, h.data(), in_bytes, hipMemcpyHostToDevice));
        uint32_t grid = uint32_t(4 * prop.multiProcessorCount);

        Descriptor2D packed{ Merge{ C }, Embed{ C, 1u}};
        Descriptor2D padded{ Merge{C}, Embed{lda, 1u}};

        // Correctness on teh padded layout
        copy_through_desc<<<grid, kBlock>>>(dpad, din, padded, Nvis);
        HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(outp.data(), dpad, pad_bytes, hipMemcpyDeviceToHost));

        bool ok = true;
        for (uint32_t r = 0; r < R && ok; r++) {
           for (uint32_t c = 0; c < C; c++) {
               if (outp[size_t(r) * lda + c] != h[size_t(r) * C + c]) {
                   ok = false;
                   break;
               }
           }
        }

        std::printf("Descriptor (Merge+Embed) into padded buffer: %s\n", ok ? "correct" : "wrong");

        // Cost of padding
        size_t useful = size_t(Nvis) * sizeof(float);
        double gp = measure_gbs([&] -> void {copy_through_desc<<<grid, kBlock>>>(dpack, din, packed, Nvis);}, useful, 50);
        double gd = measure_gbs([&] -> void {copy_through_desc<<<grid, kBlock>>>(dpad, din, packed, Nvis);}, useful, 50);

        std::printf("         packed (lda=%u): %.1f GB/s | padded (lda=%u): %.1f GB/s\n",
                    C, gp, lda, gd);
        std::printf("         minnesoverhead (memory overhead) av padding: %.1f%%\n",
                    100.0 * (double(lda) / double(C) - 1.0));
        std::printf("         -> padding kostar lite bandbredd/minne, ger justerade rader.\n");
        std::printf("            (padding costs a little bandwidth/memory, buys aligned rows.)\n");

        HIP_CHECK(hipFree(din)); HIP_CHECK(hipFree(dpack)); HIP_CHECK(hipFree(dpad));
    }

    std::printf("Read trasforms.html and descriptors.html in cktile documentation");
    return 0;
}
