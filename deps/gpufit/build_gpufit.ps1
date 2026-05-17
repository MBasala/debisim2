# Build pygpufit's Gpufit.dll for a specific CUDA compute capability and
# install it into the active venv's pygpufit package.
#
# The shipped wheel ships a DLL built for older arches (sm_60..sm_89) which
# fails on newer GPUs with:
#     RuntimeError: status = -1, message = no kernel image is available
#                                          for execution on the device
#
# This script:
#   1. Ensures the Gpufit_build/ source tree is present alongside the venv.
#   2. Patches the legacy CMakeLists.txt to add an EXPLICIT_CUDA_ARCH_FLAGS
#      escape hatch (the bundled CUDA_SELECT_NVCC_ARCH_FLAGS knows nothing
#      about Hopper sm_90 / Blackwell sm_120).
#   3. Configures and builds Gpufit with the requested arch.
#   4. Drops the rebuilt Gpufit.dll into the venv's pygpufit package.
#
# Usage (run from repo root):
#     # Detect this machine's GPU arch automatically
#     deps\gpufit\build_gpufit.ps1
#
#     # Or specify it explicitly (e.g. 120 = Blackwell, 90 = Hopper, 89 = Ada,
#     #                                86 = Ampere consumer, 80 = A100)
#     deps\gpufit\build_gpufit.ps1 -ComputeCap 120
#
#     # Build for a different venv
#     deps\gpufit\build_gpufit.ps1 -VenvPath C:\path\to\.venv
[CmdletBinding()]
param(
    [string]$ComputeCap = "",
    [string]$BuildDir = "deps\Gpufit_build",
    [string]$VenvPath = ".venv",
    [string]$Generator = "Visual Studio 17 2022",
    [string]$Arch = "x64"
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
if (-not (Test-Path $BuildDir)) {
    throw "Build dir not found: $BuildDir.  Clone the Gpufit source tree there first."
}
if (-not (Test-Path $VenvPath)) {
    throw "Venv not found: $VenvPath"
}
$pygpufitDir = Join-Path $VenvPath "Lib\site-packages\pygpufit"
if (-not (Test-Path $pygpufitDir)) {
    throw "pygpufit not installed in venv: $pygpufitDir.  Run pip install pygpufit first."
}

# ---------------------------------------------------------------------------
# 2. Auto-detect GPU compute capability if not specified
# ---------------------------------------------------------------------------
if (-not $ComputeCap) {
    $rawCap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>&1 | Select-Object -First 1).Trim()
    if (-not $rawCap) {
        throw "Could not detect GPU compute capability via nvidia-smi.  Pass -ComputeCap explicitly."
    }
    # nvidia-smi returns e.g. "12.0" — convert to "120"
    $ComputeCap = $rawCap.Replace(".", "")
    Write-Output "Auto-detected GPU compute capability: $rawCap -> sm_$ComputeCap"
}

$archFlags = "-gencode=arch=compute_${ComputeCap},code=sm_${ComputeCap} -gencode=arch=compute_${ComputeCap},code=compute_${ComputeCap}"
Write-Output "Will build with: $archFlags"

# ---------------------------------------------------------------------------
# 3. Patch CMakeLists.txt to add EXPLICIT_CUDA_ARCH_FLAGS escape hatch
# ---------------------------------------------------------------------------
$cmakeFile = Join-Path $BuildDir "Gpufit\CMakeLists.txt"
if (-not (Test-Path $cmakeFile)) {
    throw "Cannot find $cmakeFile"
}

$content = Get-Content $cmakeFile -Raw
if ($content -notmatch "EXPLICIT_CUDA_ARCH_FLAGS") {
    Write-Output "Patching $cmakeFile to add EXPLICIT_CUDA_ARCH_FLAGS escape hatch ..."

    # Insert the new cache variable just after find_package(CUDA)
    $insertAfter = 'set( CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCH} CACHE STRING'
    $newCache = @"
# Escape hatch for modern GPUs (Hopper sm_90, Ada sm_89, Blackwell sm_120, ...)
# that the legacy CUDA_SELECT_NVCC_ARCH_FLAGS does not understand.
set( EXPLICIT_CUDA_ARCH_FLAGS "" CACHE STRING
  "Explicit nvcc -gencode flags; overrides CUDA_ARCHITECTURES when non-empty" )

$insertAfter
"@
    $content = $content.Replace($insertAfter, $newCache)

    # Bypass CUDA_SELECT_NVCC_ARCH_FLAGS when EXPLICIT_CUDA_ARCH_FLAGS is set
    $oldSelect = @"
CUDA_SELECT_NVCC_ARCH_FLAGS( code_generation_flags "`${CUDA_ARCHITECTURES}" )
list( APPEND CUDA_NVCC_FLAGS `${code_generation_flags} )
"@
    $newSelect = @"
if( EXPLICIT_CUDA_ARCH_FLAGS )
  separate_arguments( _explicit_arch_list NATIVE_COMMAND
                      "`${EXPLICIT_CUDA_ARCH_FLAGS}" )
  list( APPEND CUDA_NVCC_FLAGS `${_explicit_arch_list} )
  set( code_generation_flags `${_explicit_arch_list} )
  message( STATUS "Using EXPLICIT_CUDA_ARCH_FLAGS, bypassing arch select" )
else()
  CUDA_SELECT_NVCC_ARCH_FLAGS( code_generation_flags "`${CUDA_ARCHITECTURES}" )
  list( APPEND CUDA_NVCC_FLAGS `${code_generation_flags} )
endif()
"@
    $content = $content.Replace($oldSelect, $newSelect)
    Set-Content -Path $cmakeFile -Value $content -NoNewline
    Write-Output "Patched."
} else {
    Write-Output "CMakeLists.txt already patched, skipping."
}

# ---------------------------------------------------------------------------
# 4. Configure (clean build dir)
# ---------------------------------------------------------------------------
$buildOut = Join-Path $BuildDir "build_local"
if (Test-Path $buildOut) {
    Remove-Item -Recurse -Force $buildOut
}
New-Item -ItemType Directory -Path $buildOut -Force | Out-Null

$buildOutAbs = (Resolve-Path $buildOut).Path
$buildSrcAbs = (Resolve-Path $BuildDir).Path

Write-Output "Configuring Gpufit ..."
& cmake -S $buildSrcAbs -B $buildOutAbs `
    -G $Generator -A $Arch `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_POLICY_DEFAULT_CMP0146=OLD `
    -DCMAKE_POLICY_DEFAULT_CMP0148=OLD `
    -DEXPLICIT_CUDA_ARCH_FLAGS="$archFlags" `
    -DUSE_CUBLAS=OFF
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

# ---------------------------------------------------------------------------
# 5. Build (Gpufit target only — skip examples/tests/bindings)
# ---------------------------------------------------------------------------
Write-Output "Building Gpufit ..."
& cmake --build $buildOutAbs --target Gpufit --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

# ---------------------------------------------------------------------------
# 6. Install into the venv (back up the old DLL first)
# ---------------------------------------------------------------------------
$newDll = Join-Path $buildOut "Release\Gpufit.dll"
$targetDll = Join-Path $pygpufitDir "Gpufit.dll"
$backupDll = "$targetDll.bak"

if ((Test-Path $targetDll) -and -not (Test-Path $backupDll)) {
    Copy-Item $targetDll $backupDll
    Write-Output "Backed up original DLL -> $backupDll"
}
Copy-Item $newDll $targetDll -Force
Write-Output "Installed $newDll -> $targetDll"
Write-Output "Done.  sm_$ComputeCap Gpufit.dll active in $VenvPath."
