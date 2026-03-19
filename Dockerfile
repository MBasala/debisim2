# ===========================================================================
# DEBISim2 Docker Image
#
# Single-stage build using pre-built gpufit wheel (custom COMPTON_PE +
# MATERIAL_BASIS models compiled locally, copied into the image).
#
# NOTE: The gpufit wheel must be pre-built for Linux before building
#       this image. See deps/gpufit/README.md for build instructions.
#       Place the wheel at: docker/pyGpufit-linux.whl
#
# Usage:
#   docker build -t debisim2 .
#   docker run --gpus all -v $(pwd)/results:/app/results debisim2 \
#       --config configs/config_calibration_phantom_dect.py \
#       --sim_dir results/phantom/ --num_bags 1
#
# Requires: NVIDIA Container Toolkit (nvidia-docker2)
# ===========================================================================

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

# Install uv binary directly (avoids PEP 668 restriction)
ADD https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz /tmp/uv.tar.gz
RUN tar -xzf /tmp/uv.tar.gz -C /usr/local/bin --strip-components=1 && \
    rm /tmp/uv.tar.gz && \
    chmod +x /usr/local/bin/uv

# Create venv (all deps install here, not system Python)
RUN uv venv /app/.venv
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency specification first (cache layer)
COPY pyproject.docker.toml /app/pyproject.toml

# Copy pre-built gpufit wheel (custom COMPTON_PE + MATERIAL_BASIS models)
# If no Linux wheel exists yet, the build will still succeed without it
# (gpufit features will be unavailable at runtime).
COPY docker/ /app/docker/

# Install Python dependencies into the venv
RUN uv pip install --no-cache \
        torch torchvision --index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install --no-cache \
        astra-toolbox \
        pyyaml tabulate scikit-image scikit-learn scipy \
        trimesh tqdm matplotlib pydicom astropy psutil brt \
        scipy-stubs pytest pytest-mock && \
    ( ls /app/docker/*.whl 2>/dev/null && \
      uv pip install --no-cache /app/docker/*.whl || true ) && \
    rm -rf /app/docker

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

# Ensure venv is on PATH for non-root user
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
