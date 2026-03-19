"""
RunPod Serverless handler for DEBISim2p5.

Accepts simulation jobs via the RunPod serverless API and runs the
DEBISim pipeline, streaming per-bag progress updates back to the caller.

Input schema:
{
    "config":      str   — config file path (default: calibration phantom)
    "sim_dir":     str   — output directory (default: results/simulation/)
    "num_bags":    int   — number of bags to simulate (default: 1)
    "num_workers": int   — parallel workers (default: 1)
}
"""

import os
import time
import importlib.util as config_loader
import warnings

warnings.filterwarnings("ignore")

import runpod

# Pre-import heavy modules at cold start so they're cached for all jobs
import torch
from src.debisim_dataset_generator import run_xray_dataset_generator


def handler(job):
    """RunPod serverless handler — runs DEBISim pipeline and yields progress."""
    job_input = job["input"]

    config_file = job_input.get(
        "config", "configs/config_calibration_phantom_dect.py"
    )
    sim_dir = job_input.get("sim_dir", "results/simulation/")
    num_bags = int(job_input.get("num_bags", 1))
    num_workers = int(job_input.get("num_workers", 1))

    # ---- Validate inputs ----------------------------------------------------
    if not os.path.isfile(config_file):
        yield {"status": "error", "message": f"Config not found: {config_file}"}
        return

    if num_bags < 1:
        yield {"status": "error", "message": "num_bags must be >= 1"}
        return

    # ---- Load config --------------------------------------------------------
    spec = config_loader.spec_from_file_location("config.params", config_file)
    config = config_loader.module_from_spec(spec)
    spec.loader.exec_module(config)

    config.params["sim_dir"] = sim_dir
    config.params["num_bags"] = range(1, num_bags + 1)
    config.params["num_workers"] = num_workers

    # ---- Report GPU info ----------------------------------------------------
    gpu_name = "CPU-only"
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem // (1024 ** 2)

    yield {
        "status": "started",
        "config": config_file,
        "num_bags": num_bags,
        "num_workers": num_workers,
        "gpu": gpu_name,
        "gpu_memory_mb": gpu_mem,
        "cuda_version": torch.version.cuda or "N/A",
    }

    # ---- Run pipeline -------------------------------------------------------
    os.makedirs(sim_dir, exist_ok=True)
    t0 = time.time()

    try:
        run_xray_dataset_generator(**config.params)
    except Exception as e:
        yield {"status": "error", "message": str(e)}
        return

    elapsed = time.time() - t0

    # ---- Collect output paths -----------------------------------------------
    output_dirs = []
    if os.path.isdir(sim_dir):
        output_dirs = sorted(
            d for d in os.listdir(sim_dir)
            if os.path.isdir(os.path.join(sim_dir, d))
        )

    yield {
        "status": "completed",
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": sim_dir,
        "bags_generated": len(output_dirs),
        "output_dirs": output_dirs,
    }


runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
