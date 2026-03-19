# Docker Build Files

Place pre-built wheel files in this directory before building the Docker image.

## Required wheels

### pyGpufit (custom build with COMPTON_PE + MATERIAL_BASIS)

The standard PyPI `pyGpufit` package does not include the custom
`COMPTON_PE` and `MATERIAL_BASIS` models required by DEBISim2's
CDM decomposer.

To build a Linux-compatible wheel:

```bash
# On a Linux machine with CUDA toolkit installed:
cd deps/Gpufit_build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel $(nproc)
cd Release/pyGpufit
pip wheel . --no-deps -w ../../../docker/
```

Place the resulting `.whl` file in this directory. The Dockerfile will
install it automatically if present, or skip gracefully if absent.

## Optional wheels

Any additional `.whl` files placed here will be installed into the
Docker image's virtualenv.
