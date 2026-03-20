# ===========================================================================
# DEBISim2p5 Docker Image
#
# Multi-stage build:
#   Stage 1 (gpufit-builder): compile gpufit from source with CUDA
#   Stage 2 (deps-builder):   install all Python deps into a venv via uv
#   Stage 3 (runtime):        slim CUDA base image + venv + app code
#
# Build args:
#   CUDA_VERSION  - CUDA version tag (default: 12.6.3)
#   TORCH_INDEX   - PyTorch wheel index (default: cu128)
#
# Usage:
#   # Default: CUDA 12.6 base + PyTorch cu128 (supports Ampere through Blackwell):
#   docker compose build
#
#   # CUDA 12.6 only (no Blackwell/RTX 50-series support):
#   docker compose build --build-arg TORCH_INDEX=cu126
#
# Requires: NVIDIA Container Toolkit (nvidia-docker2)
# ===========================================================================

ARG CUDA_VERSION=12.6.3

# ---- Stage 1: Build gpufit from source ------------------------------------
# Builder needs CUDA 12.8 devel for sm_100 (Blackwell) support.
# Runtime still uses 12.6 base — PyTorch cu128 bundles its own CUDA libs.
ARG GPUFIT_CUDA_VERSION=12.8.1
FROM nvidia/cuda:${GPUFIT_CUDA_VERSION}-devel-ubuntu24.04 AS gpufit-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y -qq \
        cmake python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY deps/Gpufit_build /build/src

WORKDIR /tmp/gpufit_build

# Gpufit uses legacy FindCUDA/CUDA_SELECT_NVCC_ARCH_FLAGS which doesn't
# know sm_100 (Blackwell).  Replace the broken call with a direct set.
RUN sed -i \
        's/CUDA_SELECT_NVCC_ARCH_FLAGS( code_generation_flags "${CUDA_ARCHITECTURES}" )/set( code_generation_flags "" )/' \
        /build/src/Gpufit/CMakeLists.txt && \
    cmake /build/src \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DCUDA_NVCC_FLAGS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100"

RUN cmake --build . --target Gpufit --parallel $(nproc)

RUN cp /tmp/gpufit_build/Gpufit/libGpufit.so /build/src/Gpufit/python/pygpufit/ && \
    cd /build/src/Gpufit/python && \
    python3 -m pip wheel . --no-deps -w /wheels/

# ---- Stage 2: Install Python deps into a venv ----------------------------
ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04 AS deps-builder

ARG TORCH_INDEX=cu128
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Install uv (only needed in this stage)
ADD https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz /tmp/uv.tar.gz
RUN tar -xzf /tmp/uv.tar.gz -C /usr/local/bin --strip-components=1 && \
    rm /tmp/uv.tar.gz

# Create venv
RUN uv venv /app/.venv
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="/app/.venv/bin:$PATH"

# Copy gpufit wheel from stage 1
COPY --from=gpufit-builder /wheels/ /tmp/wheels/

# Install all Python dependencies
RUN uv pip install --no-cache \
        torch --index-url https://download.pytorch.org/whl/${TORCH_INDEX} && \
    uv pip install --no-cache \
        astra-toolbox \
        pyyaml tabulate scikit-image scikit-learn scipy \
        trimesh tqdm matplotlib pydicom astropy psutil brt \
        scipy-stubs && \
    uv pip install --no-cache /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# ---- Stage 3: Slim runtime image -----------------------------------------
ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Create non-root user
RUN useradd -m -s /bin/bash debisim

# Minimal runtime dependencies (no uv, no pip, no -dev)
# PyTorch cu126 wheels bundle their own CUDA runtime (libcublas, libcusparse,
# etc.) so we do NOT install system CUDA libs — avoids 15+ GB of duplicates.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Copy the fully-built venv from stage 2
COPY --from=deps-builder --chown=debisim:debisim /app/.venv /app/.venv

# Copy application code
COPY --chown=debisim:debisim lib/ /app/lib/
COPY --chown=debisim:debisim src/ /app/src/
COPY --chown=debisim:debisim configs/ /app/configs/
COPY --chown=debisim:debisim include/ /app/include/
COPY --chown=debisim:debisim deps/gpufit/ /app/deps/gpufit/
COPY --chown=debisim:debisim examples/ /app/examples/
COPY --chown=debisim:debisim benchmarks/ /app/benchmarks/
COPY --chown=debisim:debisim tests/ /app/tests/
COPY --chown=debisim:debisim run_dataset_generator.py /app/
COPY --chown=debisim:debisim handler.py /app/

# Results directory
RUN mkdir -p /app/results && chown debisim:debisim /app/results

USER debisim

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "from src.debisim_pipeline import DEBISimPipeline; print('OK')"

ENTRYPOINT ["python", "run_dataset_generator.py"]
CMD ["--config", "configs/config_calibration_phantom_dect.py", \
     "--sim_dir", "results/calibration_phantom/", \
     "--num_bags", "1"]