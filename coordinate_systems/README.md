# Coordinate Systems — Koordinatsystem (`coordinate_systems/`)

> *Vilket körfält äger vilket element, och i vilken register-plats?*
> *(Which lane owns which element, and in which register slot?)*

When a tile is computed by many lanes at once, each holding several elements in
registers, we need a vocabulary to reason about the mapping. Three coordinate
spaces answer the question:

| Symbol | Meaning | Question |
|---|---|---|
| **X** | WHAT element | Logical position in the tile (row, col) |
| **P** | WHO owns it | The lane / partition (thread index) |
| **Y** | WHERE in that lane | The register slot within a lane |

The **distribution** is the map `(P, Y) -> X`. It stacks on the descriptor-based
embedding from earlier parts:

```
(P, Y) --distribution--> X --Embed (descriptor)--> byte offset in global memory
```

---

## Memory Hierarchy — Minnes hierarki

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GLOBAL MEMORY (HBM)                             │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐   │
│  │ X0│ X1│ X2│ X3│ X4│ X5│ X6│ X7│ ...                              │   │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘   │
│   ↑   ↑   ↑   ↑                                                         │
│   │   │   │   └─ P=3, Y=0  (lane 3, slot 0 owns element X3)            │
│   │   │   └───── P=2, Y=0  (lane 2, slot 0 owns element X2)            │
│   │   └───────── P=1, Y=0  (lane 1, slot 0 owns element X1)            │
│   └───────────── P=0, Y=0  (lane 0, slot 0 owns element X0)            │
└─────────────────────────────────────────────────────────────────────────┘
                              │  distribution (P,Y)->X
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       WAVEFRONT (64 lanes)                              │
│                                                                         │
│  ┌──────────┬──────────┬──────────┬──────────┬───────────────┐         │
│  │ Lane P=0 │ Lane P=1 │ Lane P=2 │ Lane P=3 │   ... P=63    │         │
│  │          │          │          │          │               │         │
│  │ VGPR[Y=0]│ VGPR[Y=0]│ VGPR[Y=0]│ VGPR[Y=0]│  VGPR[Y=0]    │         │
│  │  holds X0│  holds X1│  holds X2│  holds X3│  holds X63     │         │
│  │          │          │          │          │               │         │
│  │ VGPR[Y=1]│ VGPR[Y=1]│ VGPR[Y=1]│ VGPR[Y=1]│  VGPR[Y=1]    │         │
│  │  holds X64│ holds X65│ holds X66│ holds X67│  holds X128    │         │
│  │          │          │          │          │               │         │
│  │ VGPR[Y=2]│ VGPR[Y=2]│ VGPR[Y=2]│ VGPR[Y=2]│  VGPR[Y=2]    │         │
│  │  holds X128│ holds X129│ holds X130│ holds X131│ holds X192 │         │
│  │          │          │          │          │               │         │
│  │ VGPR[Y=3]│ VGPR[Y=3]│ VGPR[Y=3]│ VGPR[Y=3]│  VGPR[Y=3]    │         │
│  │  holds X192│ holds X193│ holds X194│ holds X195│ holds X255 │         │
│  └──────────┴──────────┴──────────┴──────────┴───────────────┘         │
│                                                                         │
│  Each lane: P is fixed (threadIdx.x), Y iterates (loop index)          │
│  Together (P, Y) maps to a unique X via the distribution function      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Terminology & Code Mapping — Hårdvarutermologi

This section explicitly maps the software terms in the code to the physical hardware
on the MI300X (CDNA 3). Understanding this bridge is essential for reasoning about
performance and correctness.

### Threads, Lanes, and Wavefronts

| Software Term (HIP) | Hardware Reality (MI300X) | What it means |
|---|---|---|
| **Thread** | Logical execution path | A single instance of your kernel code. It has its own VGPRs and executes instructions sequentially. |
| **Lane** | Physical hardware slot | One of 64 SIMD execution units inside a Compute Unit. A thread *occupies* a lane. |
| **Wavefront** | 64 threads running in lockstep | The fundamental scheduling unit. The GPU fetches **one instruction** and broadcasts it to **64 lanes simultaneously** (SIMD). |
| **Block** | Group of threads (e.g., 256) | A scheduling boundary. All threads in a block share the same block-level identifiers (`blockIdx`, `blockDim`). |

**In this code:** When you see `gather_fragment<<<1, 64>>>`, you are launching exactly
**one wavefront**: 64 logical threads assigned to 64 physical lanes on a single Compute Unit.
They execute the `for (Y...)` loop together, step-by-step, in perfect lockstep.

### Compute Units and XCDs

| Term | Detail | Relevance to this code |
|---|---|---|
| **CU (Compute Unit)** | ~1907 active per MI300X | Each CU contains 64 lanes, VGPR/SGPR files, L1 cache, and 64KB LDS. One thread block is anchored to one CU. |
| **XCD (eXChip Die)** | 6 compute dies per chip | The MI300X is split across 6 physical dies. Kernels are distributed across all 6 XCDs automatically. A single block never spans XCDs. |
| **`__shared__` / LDS** | 64KB per CU | Strictly local to the CU/XCD where the block runs. This code uses **zero LDS**; all data flows through caches. |

### Registers: SGPR vs VGPR

The compiler places variables in different register files based on whether their values
are identical across a wavefront or unique per lane:

| Variable in Code | Register File | Why? |
|---|---|---|
| `din`, `dout`, `n`, `V` | **SGPR** (Scalar) | Kernel arguments. Every thread in the block sees the exact same pointer/value. |
| `blockIdx.x`, `blockDim.x` | **SGPR** (Scalar) | Block configuration. Identical for all 256 threads in a block. |
| `threadIdx.x` | **VGPR** (Vector) | Unique per thread (0..255). Each lane holds its own value. |
| `P`, `X`, `Y`, `flatX` | **VGPR** (Vector) | Derived from `threadIdx.x`. Every lane computes and stores its own index. |

**Data flow:** `HBM3 → L2 Cache → L1 Cache → VGPR`. Pointers live in SGPRs to calculate
addresses; the actual float data lands in VGPRs for computation.

### Compiler Attributes & Execution Overhead

| Attribute | Who calls it? | Latency | Hardware impact |
|---|---|---|---|
| `__global__` | CPU (`<<<...>>>`) | **High** (~10-50 µs) | Triggers a full kernel launch: command packet, scheduler setup, wavefront initialization. |
| `__device__` | GPU (kernel/helper) | **Low** (~ns) | Standard on-chip function call. Often inlined at `-O3`. |
| `__host__ __device__` | CPU or GPU | **Zero** (inlined) | Compiler generates both x86 and CDNA machine code. At `-O3`, the call disappears entirely. |
| `__restrict__` | N/A (compiler hint) | N/A | Tells the compiler pointers don't overlap. Enables aggressive VGPR caching and instruction reordering. |

---

## Step-by-Step Walkthrough — Steg för steg

### STEG 1 — P only: One element per lane (`X = P`)

The simplest possible distribution. Each lane owns exactly one element, and the
element index equals the lane index. No Y dimension needed.

```
Global Memory:  [ X0 │ X1 │ X2 │ X3 │ ... │ X(N-1) ]
                    ↑    ↑    ↑    ↑          ↑
                    │    │    │    │          │
                   P=0  P=1  P=2  P=3       P=N-1

Kernel:  P = blockIdx.x * blockDim.x + threadIdx.x   // WHO
         X = P                                        // WHAT, same as WHO
         out[X] = in[X] * in[X]
```

**Launch configuration:** 4096 blocks x 256 threads = 1,048,576 lanes total.
Each lane squares one float. This is the familiar vector-square pattern.

**Key insight:** When Y has length 1, `X = P`. The vocabulary is established:
P answers "who", X answers "what".

---

### STEG 2 — Adding Y: Each lane owns V elements

Now each lane processes multiple elements. The same data can be distributed in
different ways, with dramatically different memory performance.

#### Cyclic Distribution (Coalesced)

```
X(P, Y) = Y * Pcount + P

Visual layout (V=4, Pcount=4 lanes shown):

  Y=0:  [ P0 │ P1 │ P2 │ P3 │ P0 │ P1 │ P2 │ P3 │ ... ]
  Y=1:  [ P0 │ P1 │ P2 │ P3 │ P0 │ P1 │ P2 │ P3 │ ... ]
  Y=2:  [ P0 │ P1 │ P2 │ P3 │ P0 │ P1 │ P2 │ P3 │ ... ]
  Y=3:  [ P0 │ P1 │ P2 │ P3 │ P0 │ P1 │ P2 │ P3 │ ... ]

Global memory for Y=0: [ X0 │ X1 │ X2 │ X3 │ X4 │ X5 │ X6 │ X7 │ ... ]
                        ↑    ↑    ↑    ↑
                       P0   P1   P2   P3   ← consecutive lanes, consecutive addresses → COALESCED
```

For a fixed Y, consecutive lanes access consecutive memory addresses. The memory
controller merges these into a single cache-line fetch. **This is the grid-stride
pattern you've always used.**

#### Blocked Distribution (Strided / Uncoalesced)

```
X(P, Y) = P * V + Y

Visual layout (V=4, Pcount=4 lanes shown):

  Y=0:  [ P0 │       │       │       │ P1 │       │       │       │ ... ]
  Y=1:  [       │ P0  │       │       │       │ P1  │       │       │ ... ]
  Y=2:  [       │       │ P0  │       │       │       │ P1  │       │ ... ]
  Y=3:  [       │       │       │ P0  │       │       │       │ P1  │ ... ]

Global memory: [ X0 │ X1 │ X2 │ X3 │ X4 │ X5 │ X6 │ X7 │ ... ]
                 ↑           ↑           ↑           ↑
                P0          P0          P1          P1
                Y=0         Y=3         Y=0         Y=3

Consecutive lanes (P0, P1) access addresses V apart → UNCOALESCED
```

For a fixed Y, consecutive lanes are V elements apart in memory. Each lane
touches a different cache line. Memory throughput degrades proportionally.

**Key insight:** Same data, same amount of work — the `(P, Y) -> X` distribution
decides the speed. *Samma data, fördelningen avgör farten.*

---

### STEG 3 — 2-D Fragment: `(P, Y) -> (row, col)`

A 16x16 tile (256 elements) distributed across 64 lanes, each holding 4 register
slots. This is the shape of an MFMA fragment used in tile-based matrix
multiplication.

```
Tile (16x16 = 256 elements):                    Wavefront (64 lanes x 4 slots):

  (row, col)                                     Lane 0    Lane 1    Lane 2    ...   Lane 63
  ┌──────────────────────────┐                   ┌───────┐ ┌───────┐ ┌───────┐       ┌───────┐
  │ X0   X1   X2   ...  X15  │                   │ Y=0   │ │ Y=0   │ │ Y=0   │       │ Y=0   │
  │ X16  X17  X18  ...  X31  │                   │ ←X0   │ │ ←X1   │ │ ←X2   │       │ ←X63  │
  │ ...                      │                   ├───────┤ ├───────┤ ├───────┤       ├───────┤
  │ X240 X241 ...  X255      │                   │ Y=1   │ │ Y=1   │ │ Y=1   │       │ Y=1   │
  └──────────────────────────┘                   │ ←X64  │ │ ←X65  │ │ ←X66  │       │ ←X127 │
                                                 ├───────┤ ├───────┤ ├───────┤       ├───────┤
Distribution:                                   │ Y=2   │ │ Y=2   │ │ Y=2   │       │ Y=2   │
  flatX = Y * 64 + P                            │ ←X128 │ │ ←X129 │ │ ←X130 │       │ ←X191 │
  row   = flatX / 16                            ├───────┤ ├───────┤ ├───────┤       ├───────┤
  col   = flatX % 16                            │ Y=3   │ │ Y=3   │ │ Y=3   │       │ Y=3   │
                                                 │ ←X192 │ │ ←X193 │ │ ←X194 │       │ ←X255 │
Example for Lane 0:                             └───────┘ └───────┘ └───────┘       └───────┘
  Y=0: flatX=0,  row=0, col=0  → tile[0][0]
  Y=1: flatX=64, row=4, col=0  → tile[4][0]
  Y=2: flatX=128,row=8, col=0  → tile[8][0]
  Y=3: flatX=192,row=12,col=0  → tile[12][0]

Each lane gathers 4 elements from the tile into its VGPR slots.
```

**Key insight:** This is a representative distribution showing the *shape* of how
MFMA fragments are laid out. The exact register layout for `v_mfma_f64_16x16x4`
is derived from the ISA (covered in Part 8). This kernel demonstrates the
*vocabulary*, not the final answer. *Formen på frågan, inte det exakta svaret.*

---

## Build & Run — Bygg och kör

```bash
# Direct compilation (arch-neutral fat binary)
amdclang++ -std=c++20 -O3 -x hip \
    --offload-arch=gfx942 --offload-arch=gfx950 \
    main.hip -o coords

# Or via CMake
cmake --build build/release --target coordinate_systems
./build/release/coordinate_systems
```

### Expected output

```
running on: gfx942

[STEG 1] P-only (X = P): KORREKT
[STEG 2] cyclic & blocked both correct copies: KORREKT
         cyclic  (coalesced): ~1800.0 GB/s
         blocked (strided)  : ~400.0 GB/s
         -> samma data, fördelningen avgör farten (4.50x).
            (same data; the (P,Y)->X distribution decides the speed.)
[STEG 3] fragment gather (P,Y)->(row,col): KORREKT

Klart. Läs nu coordinate_systems.html.
```

The bandwidth ratio between cyclic and blocked will vary by hardware, but expect
a **3–6x difference** on CDNA3/CDNA4. The cyclic pattern achieves near-peak HBM
bandwidth; the blocked pattern is limited by cache-line utilization.

---

## What This Teaches — Vad detta lär

1. **Vocabulary first.** Before optimizing, name the spaces: P (who), Y (where),
   X (what). Clear names prevent bugs in complex tile distributions.

2. **Distribution is performance.** Two kernels that compute identical results can
   differ by 5x in throughput purely because of how `(P, Y) -> X` is defined.

3. **Cyclic = coalesced = fast.** The grid-stride loop you've been writing is
   actually a cyclic distribution. Now you have the formal name for it.

4. **Fragment shape precedes ISA detail.** Understanding `(P, Y) -> (row, col)`
   mapping is the prerequisite for understanding MFMA register layouts. Get the
   coordinate transformation right first; the ISA encoding follows.

---

## Connection to Other Parts — Koppling till andra delar

| Part | Connection |
|---|---|
| Part 2: Views | The Embed step: `X -> byte offset` uses tensor descriptors |
| Part 4: Tile Distribution | Extends `(P, Y) -> X` to multi-warp tile layouts |
| Part 8: MFMA ISA | The exact register layout for `v_mfma_f64_16x16x4` |

*Detta är det branta steget — utan koordinatsystem blir resten oklart.*
*(This is the steep step — without coordinate systems, everything else is unclear.)*
