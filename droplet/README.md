# Droppe-uppsättning — Droplet Provisioning (`hip-learn/droplet/`)
# From bare DigitalOcean VM → working Zed kernel-dev session, every morning

*This directory holds everything needed to stand up a **clean droplet** daily. The
guiding split: your **repo travels** (configs committed to git), the **system layer is
ephemeral** (reinstalled per droplet). Two scripts, one reboot between them.*

---

## 1. Where it lives — Var det bor (repo tree)

Commit all of this to `hip-learn` so a fresh `git clone` carries it onto every new VM:

```
hip-learn/
├── CMakeLists.txt
├── CMakePresets.json            # fat-binary gfx942;gfx950, debug + release
├── .clangd                      # strips --offload-arch for clangd's single frontend
├── .zed/
│   ├── settings.json            # clangd → /usr/bin/clangd + query-driver
│   ├── tasks.json               # build / run / profile / isa / bootstrap tasks
│   └── debug.json               # rocgdb (DAP) configs
│
├── droplet/                     # ← provisioning lives here
│   ├── 1-install-rocm-7.2.4.sh  # your guide, steps 2–6 + reboot checkpoint
│   ├── 2-bootstrap-droplet.sh   # dev env: ninja, clangd, venv + pip deps, cmake
│   ├── README.md                # this file
│   └── venv/                    # python venv (gitignored, created by bootstrap)
│
├── 00-toolchain-smoke/
│   └── toolchain_smoke.hip
├── 01-buffer-and-tensor-views/
│   └── 01-buffer-and-tensor-views.hip
└── …                            # Parts 3–8 as you go
```

The numeric prefixes (`1-`, `2-`) encode run order. *Siffrorna är körordningen* (the numbers
are the run order).

> **One architectural caveat — en arkitektonisk reservation:** the ROCm installer is
> **droplet-scoped**, not project-scoped — it provisions the *machine*, and every project
> (inference-ops, hip-learn) needs the same machine. Living in `hip-learn/droplet/` is fine
> for now, but if a second project starts depending on it, promote `1-install-rocm-7.2.4.sh`
> to a shared dotfiles/ops repo and have `hip-learn` reference it, rather than duplicating.
> *Maskinnivå hör egentligen hemma på maskinnivå* (machine-level work really belongs at the
> machine level). Not urgent today; flagging it so it doesn't ossify silently.

---

## 2. The daily order — Den dagliga ordningen

There is a **reboot in the middle** (your guide's Step 7). That divides the morning into
**three runs**, not one. *En omstart delar morgonen i tre körningar.*

### Run A — install ROCm (on a bare droplet)

```bash
git clone <your-hip-learn-url> && cd hip-learn
chmod +x droplet/*.sh                 # first time only
./droplet/1-install-rocm-7.2.4.sh     # steps 2–6, then stops at the DKMS checkpoint
```

Eyeball the `dkms status` it prints. När du är nöjd (when satisfied):

```bash
reboot
```

> If the droplet image **already** has a working ROCm (driver bound), Run A's health gate
> detects it via `rocminfo | grep gfx9`, prints *"already healthy"*, and exits without
> touching anything — so it's safe to run unconditionally. *Säker att alltid köra.*

### Run B — verify, after reboot

```bash
cd hip-learn
./droplet/1-install-rocm-7.2.4.sh     # gate now passes → "✓ ROCm already healthy"
```

That same script is your verification step — if it says *healthy*, the driver bound and the
runtime works. *Samma skript bekräftar — grönt ljus.* (If it instead tries to reinstall, the
driver did **not** bind; check `dmesg | grep -i amdgpu` for the TRN/PF errors from your guide.)

### Run C — dev environment + Zed

```bash
./droplet/2-bootstrap-droplet.sh      # ninja, clangd, python venv + pip deps, cmake configure
# then connect Zed: Remote Projects → gpu-droplet → open /root/hip-learn
```

The bootstrap creates a Python venv at `droplet/venv/` (gitignored) and installs the
rocprof-compute dependencies there. To use the venv manually later:

```bash
source droplet/venv/bin/activate
```

---

## 3. The whole thing as a glance — Hela flödet på en rad

```
bare VM ─▶ clone ─▶ [1] install-rocm ─▶ REBOOT ─▶ [1] verify ─▶ [2] bootstrap ─▶ connect Zed
          (repo)     (steps 2–6)                   (gate)        (dev layer)
```

Two scripts, one reboot, one editor connection. *Två skript, en omstart, ett verktyg.*

---

## 4. Script internals — Hur skripten fungerar

**`2-bootstrap-droplet.sh`** resolves paths relative to its own location (`droplet/`), so
it works regardless of where you invoke it from. CMake runs in a subshell that `cd`s to the
repo root, keeping the parent shell's working directory unchanged.

The Python venv lives at `droplet/venv/` and is excluded from git via `.gitignore`. The
script installs `python3-venv` and `python3-dev` system packages if missing, creates the
venv, activates it, and installs rocprof-compute dependencies.

**(a)** Update the **"bootstrap droplet"** task in `.zed/tasks.json` to the new path:

```jsonc
{ "label": "bootstrap droplet",
  "command": "./droplet/2-bootstrap-droplet.sh",   // was: ./bootstrap-droplet.sh
  "cwd": "$ZED_WORKTREE_ROOT", "reveal": "always" }
```

---

## 5. Verification checklist — Kontrollista

- [ ] `droplet/` committed to git with both scripts + this README; scripts are `chmod +x`.
- [ ] **Run A** on a bare droplet installs cleanly and stops at the DKMS checkpoint.
- [ ] After `reboot`, **Run B** prints `✓ ROCm already healthy` and a `gfx950` (or `gfx942`) line.
- [ ] `amd-smi` shows the GPU bound (driver in use: `amdgpu`, not just "Kernel modules").
- [ ] **Run C** (`2-bootstrap-droplet.sh`) finishes and prints the live arch.
- [ ] Zed connects; a `.hip` file resolves headers; the **cmake build (debug)** task compiles.

---

*Klart. En ren droppe blir en arbetsmiljö med två skript och en omstart.*
*(Done. A clean droplet becomes a working environment with two scripts and one reboot.)*
