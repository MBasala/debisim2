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

        cd /build/src
        mkdir -p /tmp/gpufit_build && cd /tmp/gpufit_build

        cmake /build/src \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=OFF \
            2>&1 | tail -5

        cmake --build . --config Release --parallel $(nproc) 2>&1 | tail -5

        cd Release/pyGpufit
        python3 -m pip wheel . --no-deps --break-system-packages \
            -w /build/out/ 2>&1 | tail -3

        echo ""
        echo "=== Built wheel ==="
        ls -la /build/out/*.whl
    '

echo ""
echo "Wheel ready in docker/:"
ls -la "$SCRIPT_DIR"/*.whl 2>/dev/null || echo "  (no wheel found — check errors above)"
