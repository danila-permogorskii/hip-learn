# hip-learn — AMD GPU Programming, One Kernel at a Time

> *En öppen läroplan. Every line handwritten. No copy-paste, no shortcuts.*
> *(An open curriculum. Every line handwritten. No copy-paste, no shortcuts.)*

This is a living notebook — my path into AMD GPU programming, built from scratch on
Instinct MI300X/MI350X hardware. Each directory is a self-contained experiment: a question
I asked the hardware, a kernel I wrote to answer it, and the numbers that came back.

If you're learning HIP or CK/tile-based programming alongside me, consider this a shared
whiteboard. If you're evaluating whether I know what I'm doing with AMD GPUs — this repo
is the answer. *Detta är mitt visitkort.* (This is my visiting card.)

---

## About — Om mig

I'm **Danila Permogorsky** — CTO at [bogdanna.dev](https://bogdanna.dev). I spend my days
building systems and my evenings understanding what makes GPUs tick. This repo is where
those two worlds meet.

Reach me at **[developer.permogorsky@gmail.com](mailto:developer.permogorsky@gmail.com)** —
happy to talk AMD, GPU architecture, or anything technical.

A note on process: I use **Qwen3.6-27B**, running on our own bogdanna.dev infrastructure
at [iict.kz](https://iict.kz/ru/), as a coding assistant — catching errors, reviewing
kernels, and helping bootstrap the dev environment. The kernels are mine; the assistant
is just a very fast pair of extra eyes. *Koderna är mina; assistenten är bara extra ögon.*

---

## Philosophy — Filosofi

- **Measure, don't trust.** Spec sheets lie about virtual functions. `hipEvent` timing
  and `rocprofv3` don't. *Mät, gissa inte.*
- **Every line is mine.** No tutorial copy-paste. If a pattern looks familiar, it's because
  I studied it, rewrote it, and internalized it.
- **Errors are loud.** The `HIP_CHECK` macro is non-negotiable. Silent GPU failures are
  the enemy. *Tysta fel är fienden.*
- **The dev loop is the skill.** Building, debugging with `rocgdb`, profiling with
  `rocprofv3`, reading disassembly — that's 80% of the work. The kernel is the easy part.

---

## Hardware — Hårdvara

| Component | Detail |
|---|---|
| GPU | AMD Instinct MI300X (gfx942, CDNA3) / MI350X (gfx950, CDNA4) |
| Platform | DigitalOcean GPU droplets (SR-IOV virtual functions) |
| ROCm | 7.2.4 |
| Compiler | `amdclang++` (not the deprecated `hipcc`) |
| Build system | CMake + Ninja |

Provisioning a fresh droplet from zero is automated — see
[`droplet/`](droplet/README.md) for the two-script, one-reboot workflow.

---

## What's Here — Vad som finns här

These are unordered experiments. Each one explores a specific question about GPU programming.
The list will grow and reorganize as the learning progresses.

| Directory | Question it answers |
|---|---|
| [`00-toolchain-smoke/`](00-toolchain-smoke/) | Does the build-debug-profile-trace loop work? What HBM bandwidth does this VF actually deliver? |
| [`hip_runtime_api/`](hip_runtime_api/) | How do device initialization and memory management work at the API level? |
| [`views/`](views/) | How do buffer and tensor views map to GPU memory layouts? |
| [`TileDistribution/`](TileDistribution/) | How do tile layouts and distributions affect compute patterns? |
| [`matmul/`](matmul/) | How does matrix multiplication look when built from first principles? |
| [`sample/`](sample/) | Scratch space for quick prototypes and ideas. |

---

## Build & Run — Bygg och kör

```bash
# Configure (debug for rocgdb, release for profiling)
cmake --preset debug
cmake --preset release

# Build
cmake --build build/debug
cmake --build build/release

# Run
./build/debug/toolchain_smoke
./build/release/toolchain_smoke
```

For the full dev loop on any kernel:

```bash
rocgdb ./build/debug/<target>              # step through kernel lanes
rocprofv3 --stats -- ./build/release/<target>   # kernel stats
rocprofv3 --hip-trace --hsa-trace -- ./build/release/<target>  # full trace
```

---

## Tooling — Verktyg

- **Editor:** Zed (configs in `.zed/`). I also kept `.vscode/` for compatibility, but Zed
  is my daily driver — faster, lighter, and just feels right for GPU code.
  *Zed känns bättre för mig än VS Code.*
- **Language server:** clangd (`.clangd` strips `--offload-arch` for the CPU frontend)
- **Debugger:** rocgdb (DAP configs in `.zed/debug.json`)
- **Profilers:** rocprofv3, rocprof-compute (Python deps in `droplet/venv/`)
- **Formatting:** `.clang-format` and `.cmake-format.py` follow
  [ROCm rocm-libraries](https://github.com/ROCm/rocm-libraries) standards
  (Google style, 100 col, indent 4). clangd applies them on-save automatically.

---

## Where This Is Going — Vart det går

The long-term direction is CK/tile-based kernels — the kind of code that powers
[rocm-libraries](https://github.com/rocm/rocm-libraries) (the successor to the now-deprecated
Composable Kernel) and modern AMD high-performance libraries. Getting there means
understanding every layer below it first: memory hierarchies, wavefront behavior, LDS usage,
tile distributions, and the ISA that comes out the other end of `amdclang++`.

*Detta är en lång resa. Varje kernel är ett steg.*
*(This is a long journey. Each kernel is a step.)*

---

## License

[MIT](LICENSE) — learn from it, fork it, use it. The only request: write your own kernels
too. *Skriv dina egna kärnor.*
