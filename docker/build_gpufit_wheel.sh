#!/bin/bash
# =================================================================
# Build a Linux pygpufit wheel with custom COMPTON_PE + MATERIAL_BASIS
# models using Docker (no Linux machine needed).
#
# Usage:
#   cd docker/
#   bash build_gpufit_wheel.sh
#
# Output: docker/pyGpufit-*.whl (Linux manylinux wheel)
# =================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Building gpufit Linux wheel via Docker..."
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  Source: $PROJECT_ROOT/deps/Gpufit_build"
echo "  Output: $SCRIPT_DIR"

# Verify source exists before launching Docker
if [ ! -d "$PROJECT_ROOT/deps/Gpufit_build" ]; then
    echo "ERROR: Gpufit_build not found at $PROJECT_ROOT/deps/Gpufit_build"
    echo "If running from WSL, cd to the /mnt/c/... path first:"
    echo "  cd /mnt/c/path/to/debisim2/docker && bash build_gpufit_wheel.sh"
    exit 1
fi

docker run --rm --gpus all \
    -v "$PROJECT_ROOT/deps/Gpufit_build:/build/src:ro" \
    -v "$SCRIPT_DIR:/build/out" \
    nvidia/cuda:13.1.0-devel-ubuntu24.04 \
    bash -c '
        set -e
        apt-get update -qq && apt-get install -y -qq \
            cmake python3-dev python3-pip python3-venv > /dev/null 2>&1

        mkdir -p /tmp/gpufit_build && cd /tmp/gpufit_build

        # Force modern CUDA architectures only — CUDA 13.x dropped
        # support for sm_53 through sm_72.
        # CUDA_ARCHITECTURES is passed to CUDA_SELECT_NVCC_ARCH_FLAGS()
        # in Gpufit's CMakeLists.txt — it accepts "X.Y" format strings.
        cmake /build/src \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=OFF \
            -DCUDA_ARCHITECTURES="7.5;8.0;8.6;8.9;9.0" \
            2>&1

        # Build only the Gpufit target (skip Cpufit)
        # Show full output for debugging
        cmake --build . --target Gpufit --parallel $(nproc) 2>&1

        # On Linux with make, output is in build root (not Release/)
        # Find the pyGpufit directory
        PYDIR=$(find /tmp/gpufit_build -name "setup.py" -path "*/pyGpufit/*" -printf "%h\n" | head -1)
        if [ -z "$PYDIR" ]; then
            echo "ERROR: Could not find pyGpufit setup.py"
            echo "Build tree:"
            find /tmp/gpufit_build -type f -name "*.so" -o -name "*.py" | head -30
            exit 1
        fi

        echo "Found pyGpufit at: $PYDIR"
        cd "$PYDIR"
        python3 -m pip wheel . --no-deps --break-system-packages \
            -w /build/out/ 2>&1

        echo ""
        echo "=== Built wheel ==="
        ls -la /build/out/*.whl
    '

echo ""
echo "Wheel ready in docker/:"
ls -la "$SCRIPT_DIR"/*.whl 2>/dev/null || echo "  (no wheel found — check errors above)"
