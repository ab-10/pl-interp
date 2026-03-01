#!/bin/bash
set -e

echo "=== NVIDIA Driver Install & Verify Script ==="
echo ""

# Check if drivers are already working
if nvidia-smi &> /dev/null; then
    echo "Drivers working:"
    nvidia-smi
    exit 0
fi

echo "nvidia-smi failed. Diagnosing..."
echo ""

# Check if nvidia kernel module is loaded
echo "--- Kernel module status ---"
if lsmod | grep -q nvidia; then
    echo "nvidia module IS loaded but nvidia-smi still failing (unusual)"
    lsmod | grep nvidia
else
    echo "nvidia kernel module is NOT loaded"
fi
echo ""

# Check if driver package is installed
echo "--- Installed NVIDIA packages ---"
dpkg -l | grep nvidia | grep '^ii' || echo "(none)"
echo ""

# Check kernel headers
KVER=$(uname -r)
echo "--- Kernel: $KVER ---"
if dpkg -l | grep -q "linux-headers-$KVER"; then
    echo "Kernel headers installed: OK"
else
    echo "Kernel headers MISSING. Installing..."
    sudo apt-get update -qq
    sudo apt-get install -y "linux-headers-$KVER"
fi
echo ""

# Check DKMS status
echo "--- DKMS status ---"
dkms status 2>/dev/null || echo "dkms not found"
echo ""

# If driver package is installed, try modprobe
if dpkg -l | grep -q 'nvidia-driver'; then
    echo "Driver package is installed. Trying to load module..."
    sudo modprobe nvidia 2>&1 && {
        echo "Module loaded successfully!"
        nvidia-smi
        exit 0
    } || {
        echo "modprobe failed. Rebuilding DKMS modules..."
        DRIVER_VER=$(dpkg -l | grep 'nvidia-driver-[0-9]' | grep '^ii' | head -1 | awk '{print $3}' | cut -d'.' -f1)
        sudo dkms autoinstall 2>&1 || true
        echo ""
        echo "Trying modprobe again..."
        sudo modprobe nvidia 2>&1 && {
            echo "Module loaded after DKMS rebuild!"
            nvidia-smi
            exit 0
        } || {
            echo ""
            echo "Still failing. A reboot may be required."
            echo "Run:  sudo reboot"
            echo "Then run this script again."
        }
    }
else
    echo "No NVIDIA driver package installed. Installing nvidia-driver-590..."
    sudo apt-get update -qq
    sudo apt-get install -y "linux-headers-$KVER" nvidia-driver-590
    echo ""
    echo "Install complete. A REBOOT is required."
    echo "Run:  sudo reboot"
    echo "Then run this script again."
fi
