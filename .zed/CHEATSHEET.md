# Zed + HIP — Cheat Sheet

> *Snabbreferens för daglig GPU-kodning i Zed.*
> *(Quick reference for daily GPU coding in Zed.)*

---

## 0. Open the project

```bash
zed /root/hip-learn
```

Zed picks up `.zed/settings.json`, `.zed/tasks.json`, and `.zed/debug.json` automatically.
clangd starts and indexes the repo against `build/debug/compile_commands.json`.

---

## 1. Build — Bygg

**Command palette:** `Ctrl+Shift+P` → Tasks → pick a build task.

| Task | What it does | When |
|---|---|---|
| `cmake build (debug)` | `cmake --build build/debug -j` | Before debugging, or after editing code |
| `cmake build (release)` | `cmake --build build/release -j` | Before profiling or benchmarking |
| `cmake configure (debug)` | `cmake --preset debug` | Only after editing `CMakeLists.txt` or `CMakePresets.json` |

**Keyboard shortcut:** `Ctrl+Shift+P` → type the task name → Enter.

---

## 2. Debug with rocgdb — Felsök

**Debug panel:** `Ctrl+Shift+D` → pick a debug config.

| Config | Binary |
|---|---|
| `rocgdb (DAP): toolchain_smoke` | `build/debug/toolchain_smoke` |
| `rocgdb (DAP): views` | `build/debug/views` |

Both configs auto-build before launching. The workflow:

1. Set a breakpoint on a kernel line (e.g., `out[tid] = in[tid] * in[tid]`)
2. Start the debug session
3. When the kernel hits, use **Step Into** / **Step Over** to walk wavefront lanes
4. Inspect variables in the debug panel — rocgdb knows `blockIdx`, `threadIdx`, shared memory

**Key shortcuts during debug:**
| Shortcut | Action |
|---|---|
| `F5` | Start / Continue |
| `F9` | Toggle breakpoint |
| `F10` | Step over |
| `F11` | Step into |
| `Shift+F11` | Step out |

---

## 3. Profile — Profilera

Profiling happens from the terminal, not Zed. Open a terminal panel: `Ctrl+``.

**Kernel stats (quick):**
```bash
rocprofv3 --stats -- ./build/release/toolchain_smoke
```

**Full trace (HIP + HSA):**
```bash
rocprofv3 --hip-trace --hsa-trace -- ./build/release/toolchain_smoke
```

**Roofline analysis (Python GUI):**
```bash
source droplet/venv/bin/activate
rocprof-compute ./build/release/toolchain_smoke
```

This launches a Dash web app — open the URL it prints in your browser.

---

## 4. Code Navigation — Kodnavigering

clangd (configured in `.zed/settings.json`) provides:

| Action | Shortcut |
|---|---|
| Go to definition | `Ctrl+Click` or `F12` |
| Find all references | `Shift+F12` |
| Rename symbol | `F2` |
| Hover for docs | `Ctrl+K Ctrl+I` |
| Format document | `Shift+Alt+F` |
| Go to file | `Ctrl+P` |
| Go to symbol in file | `Ctrl+Shift+O` |
| Go to line | `Ctrl+G` |

clangd uses `compile_commands.json` from `build/debug/`, so it understands HIP intrinsics,
`__global__`, `__shared__`, and all the AMDGCN builtins.

---

## 5. Formatting — Formatering

The repo uses Google-style clang-format (`.clang-format`): 100 columns, indent 4, left-aligned
pointers. This matches [ROCm rocm-libraries](https://github.com/ROCm/rocm-libraries).

**Manual format:** `Shift+Alt+F` formats the current file.

**On-save:** formatting is intentionally **off** in `.zed/settings.json` — kernel code is
deliberately structured and auto-reformat can break layout. Format manually when you're done
editing a section.

---

## 6. Terminal — Terminal

| Action | Shortcut |
|---|---|
| Toggle terminal panel | `Ctrl+`` |
| New terminal | `Ctrl+Shift+`` |
| Split terminal | `Ctrl+Shift+5` |

Use the terminal for running binaries, profiling, and one-off commands. The build tasks
show output in the task output panel (`Ctrl+Shift+P` → Tasks).

---

## 7. The Dev Loop — Dev-slingan

The full cycle for a new kernel:

```
1. Write kernel in .hip file          → Zed + clangd (autocomplete, errors)
2. Build debug                        → Task: cmake build (debug)
3. Debug with rocgdb                  → Debug panel: rocgdb (DAP)
4. Build release                      → Task: cmake build (release)
5. Profile with rocprofv3             → Terminal
6. Read disassembly (optional)        → amdclang++ --save-temps → read *.s
7. Iterate                            → back to 1
```

*Detta är 80% av arbetet. Kärnan är den lätta delen.*
*(This is 80% of the work. The kernel is the easy part.)*

---

## 8. Multi-cursor & Editing — Redigering

Zed's multi-cursor is fast for kernel edits:

| Action | Shortcut |
|---|---|
| Add cursor at click | `Ctrl+Click` |
| Select next occurrence | `Ctrl+D` |
| Split selection into lines | `Shift+Alt+I` (end of each line) |
| Undo / Redo | `Ctrl+Z` / `Ctrl+Shift+Z` |
| Comment line | `Ctrl+/` |
| Move line up/down | `Shift+Alt+↑` / `Shift+Alt+↓` |

---

## 9. Common pitfalls — Vanliga fallgropar

- **clangd shows red squiggles on `__global__`** — make sure `build/debug/compile_commands.json`
  exists. Run the `cmake configure (debug)` task if you just cloned or deleted the build dir.
- **rocgdb can't find the binary** — build the debug target first. The debug configs auto-build,
  but if you changed `CMakeLists.txt` you need to reconfigure.
- **Profiling numbers are wrong** — you're profiling the debug build. Always use `build/release`.
- **`format_on_save` reformatting your kernel** — it's set to `"off"` in settings.json. If you
  changed it, revert it. Kernel layout is intentional.

---

*En ren arbetsmiljö, ett verktyg, en slinga.*
*(A clean environment, one editor, one loop.)*
