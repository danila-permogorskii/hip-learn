#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# bootstrap-droplet.sh — daglig uppsättning av en ren ROCm-droppe
#                        daily bootstrap for a clean ROCm 7.2.x droplet
#
# Context: DigitalOcean GPU droplets ship BASE ROCm (amdclang++, rocgdb,
#          rocprofv3) but NOT the dev tooling. This script installs only the
#          *ephemeral* layer — the bits that do not live in your git repo.
#          Allt som finns i repo:t (CMakePresets, .clangd, .zed/*) följer med
#          klonen; detta skript fixar bara resten.
#
# Idempotent: re-running is safe. Kör om så ofta du vill.
# Run from the repo root:  ./bootstrap-droplet.sh
# ───────────────────────────────────────────────────────────────────────────
set -euo pipefail

# --- 0. Resolve ROCm path — lös ROCm-sökvägen ------------------------------
# /opt/rocm is usually a symlink to /opt/rocm-<version>; prefer the real dir.
ROCM_PATH="$(readlink -f /opt/rocm 2>/dev/null || echo /opt/rocm)"
echo "==> ROCm at: ${ROCM_PATH}"

# --- 1. System packages — systempaket (apt) --------------------------------
# ninja-build : the generator CMakePresets.json asks for (annars: "unable to
#               find Ninja").  clangd : the language server ROCm does NOT ship
#               (Zed spawns it; without it, every .hip file is red squiggles).
# The profiler binaries are usually already present on the DO image, but apt
# is idempotent, so listing them costs nothing on a fresh-but-different image.
echo "==> Installing system packages…"
apt-get update -qq
apt-get install -y --no-install-recommends \
    ninja-build \
    clangd \
    rocprofiler-compute \
    rocprofiler-systems \
    python3-venv \
    python3-dev \
    || echo "!! some packages may already be present — continuing"

# --- 2. Python venv + deps for rocprof-compute — virtuell miljö ------------
# The .deb drops the BINARY; the analysis/roofline ENGINE (pandas, numpy,
# dash, matplotlib…) is Python and must be pip-installed separately.
# Paketet ger skalet, pip ger motorn.
# We install into a venv under droplet/venv/ to keep system Python clean.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "==> Creating Python venv at ${VENV_DIR}…"
    python3 -m venv "${VENV_DIR}"
fi

# Activate the venv for the remainder of this script
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "==> venv activated: $(which python) $(python --version 2>&1)"

REQ="${ROCM_PATH}/libexec/rocprofiler-compute/requirements.txt"
if [[ -f "${REQ}" ]]; then
    echo "==> Installing rocprof-compute Python deps into venv…"
    pip install --upgrade pip
    pip install -r "${REQ}"
else
    echo "!! ${REQ} not found — skipping rocprof-compute deps"
fi

# --- 3. (Optional) Zed proxy guard — Zed-proxyskydd ------------------------
# Zed's remote handshake travels over the SSH channel; ANY output your shell
# prints on a NON-interactive login corrupts it and Zed hangs at
# "Starting proxy…". Stock Ubuntu ~/.bashrc already guards this at the top,
# so this is only a safety net if you added noisy output above that guard.
GUARD='[[ $- != *i* ]] && return 0'
if ! grep -qF "$- != *i*" "${HOME}/.bashrc" 2>/dev/null; then
    echo "==> Adding non-interactive guard to ~/.bashrc (Zed proxy safety)…"
    printf '%s\n%s\n' "$GUARD" "$(cat "${HOME}/.bashrc" 2>/dev/null || true)" \
        > "${HOME}/.bashrc.new" && mv "${HOME}/.bashrc.new" "${HOME}/.bashrc"
else
    echo "==> ~/.bashrc already guards non-interactive shells — bra (good)"
fi

# --- 4. Configure both CMake builds — konfigurera båda byggena -------------
# debug   : the build rocgdb attaches to (-O0 -g)
# release : the build you PROFILE (-O3). Profilera aldrig en debug-build.
# This also generates build/<type>/compile_commands.json for clangd.
# The script lives in droplet/ but CMakePresets.json is at repo root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "==> Configuring CMake presets (in ${REPO_ROOT})…"
(
    cd "${REPO_ROOT}"
    cmake --preset debug
    cmake --preset release
)

# --- 5. Report the live arch this session — rapportera arkitekturen --------
# Droplets vary draw-to-draw: gfx942 (MI300X) vs gfx950 (MI350X). Always know
# which one you drew before you reason about FP64 ceilings or LDS size.
echo "==> This session's GPU:"
rocminfo | grep -m1 -i 'gfx' || echo "!! rocminfo found no gfx agent"

echo ""
echo "✓ Klart. Bootstrap complete. Connect Zed (Remote Projects → your droplet)"
echo "  and build with the 'cmake build (debug)' task."
