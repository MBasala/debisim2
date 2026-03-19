# Docker

All Docker-related files live in this directory.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Main DEBISim2 runtime image |
| `Dockerfile.gpufit` | Builds the Linux pygpufit wheel with custom models |
| `docker-compose.yml` | Services: debisim2, test, benchmark |
| `pyproject.docker.toml` | Linux-compatible Python dependencies |
| `build_gpufit_wheel.sh` | Script to build the gpufit wheel via Docker |
| `*.whl` | Pre-built wheels (not committed, generated locally) |

## Quick start

```bash
# 1. Build the gpufit wheel (one-time, ~2 min)
cd docker/
bash build_gpufit_wheel.sh

# 2. Build the runtime image
docker compose build

# 3. Run
docker compose run debisim2                                    # calibration phantom
docker compose run debisim2 --config configs/config_2_firearm_only_parallelbeam_3d_dect.py --sim_dir results/firearms/
docker compose run test                                        # pytest suite
docker compose run benchmark                                   # pipeline timing
```

## pyGpufit wheel

The standard PyPI `pyGpufit` does not include the custom `COMPTON_PE`
and `MATERIAL_BASIS` models required by DEBISim2's CDM decomposer.

`build_gpufit_wheel.sh` uses `Dockerfile.gpufit` to compile the wheel
inside a CUDA devel container — no Linux machine needed, just Docker
with GPU support.

The resulting `.whl` file is placed in this directory and automatically
picked up by `Dockerfile` on the next build. If absent, the image still
builds but the CDM decomposer will be unavailable.
