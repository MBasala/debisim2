# ===========================================================================
# DEBISim2p5 Docker Image
#
# Multi-stage build:
#   Stage 1 (gpufit-builder): compile gpufit from source with CUDA
#   Stage 2 (runtime): slim CUDA runtime image with Python + all deps
#
# Usage (from docker/ directory):
#   docker compose build
#   docker compose run debisim2p5 --config configs/config_calibration_phantom_dect.py
#
# Requires: NVIDIA Container Toolkit (nvidia-docker2)
# ===========================================================================

# ---- Stage 1: Build gpufit from source ------------------------------------
FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS gpufit-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y -qq \
        cmake python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY deps/Gpufit_build /build/src

WORKDIR /tmp/gpufit_build

# Force modern CUDA architectures only (CUDA 13.x dropped sm_53-sm_72)
RUN cmake /build/src \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DCUDA_ARCHITECTURES="7.5;8.0;8.6;8.9;9.0"

# Build only Gpufit target (skip Cpufit)
RUN cmake --build . --target Gpufit --parallel $(nproc)

# Copy built .so into pygpufit package dir and build wheel
RUN cp /tmp/gpufit_build/Gpufit/libGpufit.so /build/src/Gpufit/python/pygpufit/ && \
    cd /build/src/Gpufit/python && \
    python3 -m pip wheel . --no-deps -w /wheels/

# ---- Stage 2: Runtime image -----------------------------------------------
FROM nvidia/cuda:13.1.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Create non-root user FIRST (before any COPY --chown)
RUN useradd -m -s /bin/bash debisim

# System dependencies (minimal — no -dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        libgomp1 libcublas-13-1 libcusparse-13-1 \
        && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Install uv binary directly (avoids PEP 668 restriction)
ADD https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz /tmp/uv.tar.gz
RUN tar -xzf /tmp/uv.tar.gz -C /usr/local/bin --strip-components=1 && \
    rm /tmp/uv.tar.gz && \
    chmod +x /usr/local/bin/uv

# Create venv owned by debisim from the start
RUN uv venv /app/.venv && chown -R debisim:debisim /app/.venv
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="/app/.venv/bin:$PATH"

# Copy gpufit wheel from builder stage
COPY --from=gpufit-builder /wheels/ /tmp/wheels/

# Install Python dependencies (single layer, no cache)
RUN uv pip install --no-cache \
        torch --index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install --no-cache \
        astra-toolbox \
        pyyaml tabulate scikit-image scipy \
        trimesh tqdm matplotlib pydicom astropy psutil brt \
        scipy-stubs && \
    uv pip install --no-cache /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Copy application code — all owned by debisim (no chown layer needed)
COPY --chown=debisim:debisim lib/ /app/lib/
COPY --chown=debisim:debisim src/ /app/src/
COPY --chown=debisim:debisim configs/ /app/configs/
COPY --chown=debisim:debisim include/ /app/include/
COPY --chown=debisim:debisim deps/gpufit/ /app/deps/gpufit/
COPY --chown=debisim:debisim examples/ /app/examples/
COPY --chown=debisim:debisim benchmarks/ /app/benchmarks/
COPY --chown=debisim:debisim tests/ /app/tests/
COPY --chown=debisim:debisim run_dataset_generator.py /app/

# Results directory
RUN mkdir -p /app/results && chown debisim:debisim /app/results

# Switch to non-root user
USER debisim

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "from src.debisim_pipeline import DEBISimPipeline; print('OK')"

ENTRYPOINT ["python", "run_dataset_generator.py"]
CMD ["--config", "configs/config_calibration_phantom_dect.py", \
     "--sim_dir", "results/calibration_phantom/", \
     "--num_bags", "1"]
