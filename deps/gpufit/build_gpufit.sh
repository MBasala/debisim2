#!/usr/bin/env bash
# Build pygpufit's Gpufit shared library for a specific CUDA compute capability
# and install it into the active venv's pygpufit package.
#
# See build_gpufit.ps1 (Windows version) for the full rationale.  This script
# is the Linux/Mac equivalent — useful for RunPod / Docker deploys.
#
# Usage (run from repo root):
#     # Auto-detect this machine's GPU arch
#     bash deps/gpufit/build_gpufit.sh
#
#     # Override (e.g. 120 = Blackwell, 90 = Hopper, 89 = Ada, 86 = Ampere)
#     bash deps/gpufit/build_gpufit.sh 120
#
#     # Override venv too
#     VENV=/opt/venv bash deps/gpufit/build_gpufit.sh 120
set -euo pipefail

COMPUTE_CAP="${1:-}"
BUILD_DIR="${BUILD_DIR:-deps/Gpufit_build}"
VENV="${VENV:-.venv}"

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
[[ -d "$BUILD_DIR" ]] || {
    echo "ERROR: Build dir not found: $BUILD_DIR" >&2
    echo "       Clone the Gpufit source tree there first." >&2
    exit 1
}
[[ -d "$VENV" ]] || { echo "ERROR: Venv not found: $VENV" >&2; exit 1; }

# Locate pygpufit (Linux uses lib/, not Lib/)
PYGPUFIT_DIR=""
for cand in "$VENV/lib/python"*"/site-packages/pygpufit" \
            "$VENV/Lib/site-packages/pygpufit"; do
    if [[ -d "$cand" ]]; then PYGPUFIT_DIR="$cand"; break; fi
done
[[ -n "$PYGPUFIT_DIR" ]] || {
    echo "ERROR: pygpufit not installed in $VENV" >&2; exit 1
}

# ---------------------------------------------------------------------------
# 2. Auto-detect compute cap if not given
# ---------------------------------------------------------------------------
if [[ -z "$COMPUTE_CAP" ]]; then
    RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -n 1 | tr -d '[:space:]')
    [[ -n "$RAW" ]] || {
        echo "ERROR: nvidia-smi did not return compute_cap.  Pass it explicitly." >&2
        exit 1
    }
    COMPUTE_CAP="${RAW//./}"
    echo "Auto-detected GPU compute capability: $RAW -> sm_$COMPUTE_CAP"
fi

ARCH_FLAGS="-gencode=arch=compute_${COMPUTE_CAP},code=sm_${COMPUTE_CAP} -gencode=arch=compute_${COMPUTE_CAP},code=compute_${COMPUTE_CAP}"
echo "Will build with: $ARCH_FLAGS"

# ---------------------------------------------------------------------------
# 3. Patch CMakeLists.txt with EXPLICIT_CUDA_ARCH_FLAGS escape hatch
# ---------------------------------------------------------------------------
CMAKE_FILE="$BUILD_DIR/Gpufit/CMakeLists.txt"
[[ -f "$CMAKE_FILE" ]] || { echo "ERROR: $CMAKE_FILE missing" >&2; exit 1; }

if ! grep -q EXPLICIT_CUDA_ARCH_FLAGS "$CMAKE_FILE"; then
    echo "Patching $CMAKE_FILE ..."
    python3 - "$CMAKE_FILE" <<'PYEOF'
import sys
p = sys.argv[1]
with open(p) as f: src = f.read()

old_cache = 'set( CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCH} CACHE STRING'
new_cache = '''set( EXPLICIT_CUDA_ARCH_FLAGS "" CACHE STRING
  "Explicit nvcc -gencode flags; overrides CUDA_ARCHITECTURES when non-empty" )

''' + old_cache
src = src.replace(old_cache, new_cache, 1)

old_sel = '''CUDA_SELECT_NVCC_ARCH_FLAGS( code_generation_flags "${CUDA_ARCHITECTURES}" )
list( APPEND CUDA_NVCC_FLAGS ${code_generation_flags} )'''
new_sel = '''if( EXPLICIT_CUDA_ARCH_FLAGS )
  separate_arguments( _explicit_arch_list NATIVE_COMMAND
                      "${EXPLICIT_CUDA_ARCH_FLAGS}" )
  list( APPEND CUDA_NVCC_FLAGS ${_explicit_arch_list} )
  set( code_generation_flags ${_explicit_arch_list} )
  message( STATUS "Using EXPLICIT_CUDA_ARCH_FLAGS, bypassing arch select" )
else()
  CUDA_SELECT_NVCC_ARCH_FLAGS( code_generation_flags "${CUDA_ARCHITECTURES}" )
  list( APPEND CUDA_NVCC_FLAGS ${code_generation_flags} )
endif()'''
src = src.replace(old_sel, new_sel, 1)
with open(p, 'w') as f: f.write(src)
PYEOF
    echo "Patched."
else
    echo "CMakeLists.txt already patched, skipping."
fi

# ---------------------------------------------------------------------------
# 4. Configure + build (Gpufit target only)
# ---------------------------------------------------------------------------
BUILD_OUT="$BUILD_DIR/build_local"
rm -rf "$BUILD_OUT"
mkdir -p "$BUILD_OUT"

echo "Configuring Gpufit ..."
cmake -S "$BUILD_DIR" -B "$BUILD_OUT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_DEFAULT_CMP0146=OLD \
    -DCMAKE_POLICY_DEFAULT_CMP0148=OLD \
    -DEXPLICIT_CUDA_ARCH_FLAGS="$ARCH_FLAGS" \
    -DUSE_CUBLAS=OFF

echo "Building Gpufit ..."
cmake --build "$BUILD_OUT" --target Gpufit --config Release --parallel

# ---------------------------------------------------------------------------
# 5. Install into the venv (preserve original as .bak)
# ---------------------------------------------------------------------------
# On Linux the output is libGpufit.so, not Gpufit.dll
NEW_LIB=""
for cand in "$BUILD_OUT/Gpufit/libGpufit.so" "$BUILD_OUT/Release/Gpufit.dll"; do
    if [[ -f "$cand" ]]; then NEW_LIB="$cand"; break; fi
done
[[ -n "$NEW_LIB" ]] || { echo "ERROR: built library not found" >&2; exit 1; }

EXT="${NEW_LIB##*.}"
TARGET="$PYGPUFIT_DIR/Gpufit.$EXT"
BACKUP="$TARGET.bak"
if [[ -f "$TARGET" && ! -f "$BACKUP" ]]; then
    cp "$TARGET" "$BACKUP"
    echo "Backed up original -> $BACKUP"
fi
cp "$NEW_LIB" "$TARGET"
echo "Installed $NEW_LIB -> $TARGET"
echo "Done.  sm_$COMPUTE_CAP Gpufit active in $VENV."
