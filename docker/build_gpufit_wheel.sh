#!/bin/bash
# =================================================================
# Build a Linux pygpufit wheel with custom COMPTON_PE + MATERIAL_BASIS
# models using Docker.
#
# Usage:
#   cd docker/
#   bash build_gpufit_wheel.sh
#
# Output: docker/pyGpufit-*.whl (Linux wheel)
# =================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# If running under WSL, convert Windows-style paths to /mnt/ paths
if grep -qi microsoft /proc/version 2>/dev/null; then
    if [[ "$PROJECT_ROOT" == /c/* ]] || [[ "$PROJECT_ROOT" == /C/* ]]; then
        PROJECT_ROOT="/mnt${PROJECT_ROOT}"
        SCRIPT_DIR="$PROJECT_ROOT/docker"
        echo "WSL detected, adjusted to: $PROJECT_ROOT"
    fi
fi

# Verify source exists
if [ ! -d "$PROJECT_ROOT/deps/Gpufit_build" ]; then
    echo "ERROR: Gpufit_build not found at $PROJECT_ROOT/deps/Gpufit_build"
    echo "Run this script from the project's docker/ directory."
    exit 1
fi

echo "Building gpufit Linux wheel..."
echo "  Project: $PROJECT_ROOT"

# Build the gpufit builder image
docker build \
    -f "$SCRIPT_DIR/Dockerfile.gpufit" \
    -t gpufit-builder \
    "$PROJECT_ROOT"

# Extract the wheel
docker run --rm \
    -v "$SCRIPT_DIR:/out" \
    gpufit-builder

echo ""
echo "Wheel ready in docker/:"
ls -la "$SCRIPT_DIR"/*.whl 2>/dev/null || echo "  (no wheel found — check errors above)"
