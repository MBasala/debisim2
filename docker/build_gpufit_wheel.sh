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

docker run --rm --gpus all \
    -v "$PROJECT_ROOT/deps/Gpufit_build:/build/src:ro" \
    -v "$SCRIPT_DIR:/build/out" \
    nvidia/cuda:13.1.0-devel-ubuntu24.04 \
    bash -c '
        set -e
        apt-get update -qq && apt-get install -y -qq \
            cmake python3-dev python3-pip python3-venv > /dev/null 2>&1

        mkdir -p /tmp/gpufit_build && cd /tmp/gpufit_build

        # Only build Gpufit (GPU) target — skip Cpufit which has
        # compilation issues and is not needed for the Python wheel.
        cmake /build/src \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=OFF \
            2>&1 | tail -10

        # Build only the Gpufit target (not Cpufit)
        cmake --build . --target Gpufit --parallel $(nproc) 2>&1 | tail -10

        # On Linux with make, output is in build root (not Release/)
        # Find the pyGpufit directory
        PYDIR=$(find /tmp/gpufit_build -name "setup.py" -path "*/pyGpufit/*" -printf "%h\n" | head -1)
        if [ -z "$PYDIR" ]; then
            echo "ERROR: Could not find pyGpufit setup.py"
            find /tmp/gpufit_build -name "*.py" | head -20
            exit 1
        fi

        echo "Found pyGpufit at: $PYDIR"
        cd "$PYDIR"
        python3 -m pip wheel . --no-deps --break-system-packages \
            -w /build/out/ 2>&1 | tail -5

        echo ""
        echo "=== Built wheel ==="
        ls -la /build/out/*.whl
    '

echo ""
echo "Wheel ready in docker/:"
ls -la "$SCRIPT_DIR"/*.whl 2>/dev/null || echo "  (no wheel found — check errors above)"
