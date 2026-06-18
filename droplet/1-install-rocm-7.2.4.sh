#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# 1-install-rocm-7.2.4.sh — ROCm 7.2.4 + AMD DKMS-driver på en ren droppe
#                           ROCm 7.2.4 + AMD DKMS driver on a clean droplet
#
# Faithful to your own tested guide (steps 2–6), MI350X (gfx950) on Ubuntu 24.04.
# Trogen din egen testade guide — inga "smarta" avvikelser från det som funkar.
#
# KEY FINDING (din nyckelinsikt): install WITH the AMD DKMS driver
#   (--usecase=rocm --no-32). Do NOT use --no-dkms — the stock Ubuntu amdgpu
#   module does not bind to the MI350X VF (Fatal error / TRN_MSG_ACK / err -22).
#
# This script ends BEFORE the reboot (your Step 7) on purpose, at your Step-5
# DKMS checkpoint, so you can eyeball dkms status before rebooting.
# Run as root from anywhere:  ./1-install-rocm-7.2.4.sh
# ───────────────────────────────────────────────────────────────────────────
set -euo pipefail

# --- 0. Health gate — hälsoport (idempotency across the reboot) ------------
# rocminfo queries /dev/kfd; it only reports a gfx9 agent if the driver is
# actually BOUND. So this distinguishes "already working" from "not installed"
# AND from "user-space present but driver did not bind" (your exact failure).
# Detta skiljer "fungerar" från "ej installerat" och "drivrutin ej bunden".
if rocminfo 2>/dev/null | grep -qiE 'gfx9'; then
    echo "✓ ROCm already healthy — $(rocminfo | grep -m1 -i gfx | xargs)"
    amd-smi 2>/dev/null | head -5 || true
    echo "  Nothing to install. Run ./2-bootstrap-droplet.sh next."
    exit 0
fi
echo "==> No bound GPU detected — proceeding with install (steg 2–6)."

# Assumes root (your droplets log in as root). Antar root.
export DEBIAN_FRONTEND=noninteractive

# --- Step 2 — Base dependencies — basberoenden -----------------------------
# dkms : builds the AMD kernel module | linux-headers-$(uname -r) : MUST match
# the RUNNING kernel | build-essential : compiler | linux-firmware : blobs |
# libnuma1/numactl : needed by TransferBench bandwidth tests.
echo "==> Step 2: base dependencies…"
apt-get update
# NB: 'apt upgrade' may pull a NEW kernel. DKMS is built against the RUNNING
# kernel ($(uname -r)); the dkms hooks rebuild for the new kernel on the
# Step-7 reboot. Faithful to your guide — kept as-is.
apt-get upgrade -y
apt-get install -y \
    wget curl gnupg2 gpg ca-certificates pciutils kmod dkms \
    build-essential \
    "linux-headers-$(uname -r)" \
    "linux-modules-extra-$(uname -r)" \
    linux-firmware \
    libnuma1 numactl

# --- Step 3 — Add the official AMD ROCm 7.2.4 repo — lägg till AMD-repo -----
# NB: AMD may publish newer patch releases. This is YOUR tested-and-working
# URL; verify at https://repo.radeon.com/ before bumping the version.
echo "==> Step 3: AMD amdgpu-install repo package…"
cd /root
DEB="amdgpu-install_7.2.4.70204-1_all.deb"
wget -N "https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/${DEB}"
apt-get install -y "./${DEB}"
apt-get update

# --- Step 4 — Install ROCm 7.2.4 WITH the DKMS driver — det kritiska steget -
# THE critical line. --no-32 drops 32-bit; we deliberately OMIT --no-dkms so
# the AMD DKMS driver is built and can bind to the MI350X VF.
# Om 'amdgpu-install' inte accepterar -y på din version: ta bort -y.
echo "==> Step 4: amdgpu-install --usecase=rocm --no-32 (WITH dkms)…"
amdgpu-install -y --usecase=rocm --no-32

# --- Step 5 — DKMS checkpoint — kontrollpunkt (your verify-before-reboot) ---
echo "==> Step 5: DKMS status (verify a built/installed amdgpu entry):"
dkms status || true
echo "    amdgpu.ko location(s):"
find "/lib/modules/$(uname -r)" -name "amdgpu.ko*" 2>/dev/null | sort || true

# --- Step 6 — Rebuild initramfs — bygg om initramfs ------------------------
echo "==> Step 6: rebuilding initramfs…"
update-initramfs -u -k all

# --- Stop before Step 7 (reboot) — stanna före omstarten -------------------
cat <<'EOF'

────────────────────────────────────────────────────────────────────────────
✓ Steps 2–6 complete. Granska 'dkms status' ovan (review it above).

NEXT — gör detta för hand (do this by hand):
   1) reboot
   2) after reboot, re-run THIS script  →  it will verify and say "healthy"
   3) then run  ./2-bootstrap-droplet.sh  →  dev environment + Zed configs
────────────────────────────────────────────────────────────────────────────
EOF
