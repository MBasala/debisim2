# ===========================================================================
# DEBISim2 Docker Image
#
# Multi-stage build:
#   Stage 1 (builder): compile gpufit wheel from source
#   Stage 2 (runtime): slim CUDA image with Python + all deps
#
# Usage:
#   docker build -t debisim2 .
#   docker run --gpus all -v $(pwd)/results:/app/results debisim2 \
#       --config configs/config_calibration_phantom_dect.py \
#       --sim_dir results/phantom/ --num_bags 1
#
# Requires: NVIDIA Container Toolkit (nvidia-docker2)
# ===========================================================================

# ---- Stage 1: Build gpufit from source ------------------------------------
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04 AS gpufit-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake git python3-dev python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy gpufit source and build
COPY deps/Gpufit_build /build/Gpufit_build
WORKDIR /build/Gpufit_build

RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ARCHITECTURES="70;75;80;86;89;90" \
        -DBUILD_TESTING=OFF && \
    cmake --build . --config Release --parallel $(nproc) && \
    cd Release/pyGpufit && \
    python3 -m pip wheel . --no-deps -w /build/wheels/

# ---- Stage 2: Runtime image -----------------------------------------------
FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv python3-pip \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Create non-root user
RUN useradd -m -s /bin/bash debisim
WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency specification first (cache layer)
COPY pyproject.docker.toml /app/pyproject.toml

# Copy gpufit wheel from builder stage
COPY --from=gpufit-builder /build/wheels/ /app/wheels/

# Install Python dependencies
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install --no-cache \
        torch torchvision --index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install --no-cache \
        astra-toolbox \
        pyyaml tabulate scikit-image scikit-learn scipy \
        trimesh tqdm matplotlib pydicom astropy psutil brt \
        scipy-stubs pyGpufit pytest pytest-mock && \
    uv pip install --no-cache /app/wheels/*.whl && \
    rm -rf /app/wheels

# Copy application code
COPY lib/ /app/lib/
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY include/ /app/include/
COPY deps/gpufit/ /app/deps/gpufit/
COPY examples/ /app/examples/
COPY benchmarks/ /app/benchmarks/
COPY tests/ /app/tests/
COPY run_dataset_generator.py /app/

# Default results directory
RUN mkdir -p /app/results && chown -R debisim:debisim /app

# Switch to non-root user
USER debisim

# Activate venv in shell
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Health check — verify imports work
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "from src.debisim_pipeline import DEBISimPipeline; print('OK')"

# Default entrypoint runs the dataset generator
ENTRYPOINT ["python", "run_dataset_generator.py"]
CMD ["--config", "configs/config_calibration_phantom_dect.py", \
     "--sim_dir", "results/calibration_phantom/", \
     "--num_bags", "1"]
