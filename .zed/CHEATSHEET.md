# Zed + HIP — Repo Cheat Sheet

> *Endast det som är specifikt för detta repo. Allmänna Zed-genvägar finns i Zed-dokumentationen.*
> *(Only what's specific to this repo. General Zed shortcuts are in Zed docs.)*

---

## The Dev Loop — Dev-slingan

```
write .hip → build debug → rocgdb → build release → rocprofv3 → iterate
```

---

## Build Tasks (Zed)

Configured in `.zed/tasks.json`. Run via `Ctrl+Shift+P` → Tasks.

| Task | Command | When |
|---|---|---|
| `cmake build (debug)` | `cmake --build build/debug -j` | Before debugging or running |
| `cmake build (release)` | `cmake --build build/release -j` | Before profiling — **never profile debug** |
| `cmake configure (debug)` | `cmake --preset debug` | Only after editing `CMakeLists.txt` or `CMakePresets.json` |

---

## Debug (Zed)

Configured in `.zed/debug.json`. Open debug panel `Ctrl+Shift+D`.

| Config | Binary |
|---|---|
| `rocgdb (DAP): toolchain_smoke` | `build/debug/toolchain_smoke` |
| `rocgdb (DAP): views` | `build/debug/views` |

Both auto-build the debug target before launching. `gdb_path` points to `/opt/rocm/bin/rocgdb` —
plain gdb won't understand wavefronts.

**Adding a new target:** duplicate a config in `.zed/debug.json`, change `label` and `program`.
Also add the target to `CMakeLists.txt` first.

---

## Profile (Terminal)

Open terminal `Ctrl+``. Always use the **release** build.

```bash
# Quick kernel stats
rocprofv3 --stats -- ./build/release/<binary>

# Full HIP + HSA trace
rocprofv3 --hip-trace --hsa-trace -- ./build/release/<binary>

# Roofline GUI (requires venv)
source droplet/venv/bin/activate
rocprof-compute ./build/release/<binary>
```

---

## Disassembly — Disassembly

To read the GPU ISA that `amdclang++` produced:

```bash
# From repo root, rebuild with --save-temps
cd build/debug && ninja -v 2>&1 | grep toolchain_smoke
# Then find the .s file:
find . -name '*-gfx950.s'
```

Or manually:
```bash
/opt/rocm/llvm/bin/amdclang++ -std=c++20 -O3 --offload-arch=gfx950 -x hip \
    --save-temps <file>.hip -o /dev/null
# Produces <file>-gfx950.s
```

---

## Repo-Specific Zed Settings

In `.zed/settings.json`:

- **clangd binary** → `/usr/bin/clangd` with `--query-driver=/opt/rocm/llvm/bin/amdclang++`
  so clangd understands HIP intrinsics and `__global__`/`__shared__`
- **compile-commands-dir** → `build/debug` — if clangd shows red squiggles on HIP types,
  run the `cmake configure (debug)` task to regenerate `compile_commands.json`
- **format_on_save: "off"** — kernel layout is intentional; auto-format can break tile
  structures and LDS declarations. Format manually (`Shift+Alt+F`) when done editing.
- **language_servers: `["clangd", "!..."]`** — only clangd, no secondary analyzer.
  Two analyzers fighting over HIP code produces noise.

---

## Formatting

`.clang-format` — Google style, 100 col, indent 4, left pointers. Matches
[ROCm rocm-libraries](https://github.com/ROCm/rocm-libraries).

Format current file: `Shift+Alt+F`. Not on-save.

---

## Common Pitfalls — Vanliga fallgropar

| Symptom | Fix |
|---|---|
| Red squiggles on `__global__`, `hipMalloc`, etc. | Run `cmake configure (debug)` task — clangd needs `compile_commands.json` |
| rocgdb can't find binary | Build debug target first (or use the debug config which auto-builds) |
| Profiling gives garbage numbers | You're running the debug build. Build release. |
| `rocprof-compute` not found | Activate venv: `source droplet/venv/bin/activate` |
| clangd doesn't know AMDGCN intrinsics | Check `.zed/settings.json` — `--query-driver` must point to `amdclang++` |

---

*En slinga, ett verktyg, inget onödigt.*
*(One loop, one editor, nothing extra.)*
