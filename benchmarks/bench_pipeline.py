#!/usr/bin/env python
"""
bench_pipeline.py — Benchmark each stage of the DEBISim2 pipeline.

Runs a single bag through the full pipeline and reports per-stage
wall-clock time and peak GPU memory usage.

Usage:
    python benchmarks/bench_pipeline.py \
        --config configs/config_default_parallelbeam_3d_dect.py \
        --sim_dir results/bench_test/

Output:
    - Summary table to stdout
    - JSON file in benchmarks/results/ for tracking over time
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import argparse
import importlib.util as config_loader
from datetime import datetime

# Ensure project root is on path and is the working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def _sync_gpu():
    """Flush all pending GPU work so wall-clock timing is accurate."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _gpu_peak_mb():
    """Return peak GPU memory allocated (MB) since last reset, then reset."""
    try:
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
            return peak
    except ImportError:
        pass
    return 0.0


def _gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return 'N/A'


def _gpu_total_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e6
    except ImportError:
        pass
    return 0.0


def run_benchmark(config_path, sim_dir):
    """Run the full pipeline once, timing each stage."""

    # ------------------------------------------------------------------
    # Load config (same pattern as run_dataset_generator.py)
    # ------------------------------------------------------------------
    spec = config_loader.spec_from_file_location("config.params", config_path)
    config = config_loader.module_from_spec(spec)
    spec.loader.exec_module(config)
    params = config.params

    # Force single bag
    params['sim_dir'] = sim_dir
    params['num_bags'] = range(1, 2)

    import torch
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    from src.debisim_pipeline import DEBISimPipeline

    scanner = params['scanner']
    xray_src_mdl = params['xray_src_mdl']
    bag_creator_args = params['bag_creator_args']
    fwd_mdl_args = params.get('fwd_mdl_args', dict(
        add_poisson_noise=True, add_system_noise=True, system_gain=0.025))
    decomposer = params.get('decomposer', 'cdm')
    decomposer_args = params.get('decomposer_args', None)
    basis_fn = params.get('basis_fn', None)
    save_sino = params.get('save_sino', False)
    recon_args = params.get('recon_args', None)
    images_to_save = params.get('images_to_save',
                                ['lac_1', 'pe', 'c', 'gt'])
    slicewise = params.get('slicewise', False)
    compress_data = params.get('compress_data', False)
    dicom_output = params.get('dicom_output', False)

    bag_dir = os.path.join(sim_dir, 'simulation_001/')
    os.makedirs(bag_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialize pipeline
    # ------------------------------------------------------------------
    _sync_gpu()
    _gpu_peak_mb()  # reset
    t_init_start = time.time()

    simulator = DEBISimPipeline(
        sim_path=bag_dir,
        scanner_model=scanner,
        xray_source_model=xray_src_mdl,
        compress_data=compress_data,
    )
    simulator.mu.material('water')

    _sync_gpu()
    t_init = time.time() - t_init_start
    mem_init = _gpu_peak_mb()

    # ------------------------------------------------------------------
    # Stage 1: Bag Generator
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1: BAG GENERATOR")
    print("=" * 60)
    _sync_gpu()
    _gpu_peak_mb()
    t_bag_start = time.time()

    simulator.create_random_simulation_instance(
        bag_creator_args,
        save_images=images_to_save,
        slicewise=slicewise,
    )

    _sync_gpu()
    t_bag = time.time() - t_bag_start
    mem_bag = _gpu_peak_mb()
    print(f"  Done in {t_bag:.1f}s  (GPU peak: {mem_bag:.0f} MB)")

    # ------------------------------------------------------------------
    # Stage 2: Forward Model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: FORWARD MODEL")
    print("=" * 60)
    _sync_gpu()
    _gpu_peak_mb()
    t_fwd_start = time.time()

    simulator.run_fwd_model(**fwd_mdl_args)

    _sync_gpu()
    t_fwd = time.time() - t_fwd_start
    mem_fwd = _gpu_peak_mb()
    print(f"  Done in {t_fwd:.1f}s  (GPU peak: {mem_fwd:.0f} MB)")

    # ------------------------------------------------------------------
    # Stage 3: Decomposer (optional)
    # ------------------------------------------------------------------
    t_dec = 0.0
    mem_dec = 0.0
    if decomposer != 'none':
        print("\n" + "=" * 60)
        print("STAGE 3: DECOMPOSER")
        print("=" * 60)
        _sync_gpu()
        torch.cuda.empty_cache()
        _gpu_peak_mb()
        t_dec_start = time.time()

        simulator.run_decomposer(
            type=decomposer,
            decomposer_args=decomposer_args,
            basis_fn=basis_fn,
            save_sino=save_sino,
        )

        _sync_gpu()
        t_dec = time.time() - t_dec_start
        mem_dec = _gpu_peak_mb()
        print(f"  Done in {t_dec:.1f}s  (GPU peak: {mem_dec:.0f} MB)")

    # ------------------------------------------------------------------
    # Stage 4: Reconstructor
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 4: RECONSTRUCTOR")
    print("=" * 60)
    _sync_gpu()
    _gpu_peak_mb()
    t_recon_start = time.time()

    if recon_args is not None:
        simulator.run_reconstructor(**recon_args)
    else:
        simulator.run_reconstructor()

    _sync_gpu()
    t_recon = time.time() - t_recon_start
    mem_recon = _gpu_peak_mb()
    print(f"  Done in {t_recon:.1f}s  (GPU peak: {mem_recon:.0f} MB)")

    # ------------------------------------------------------------------
    # Stage 5: DICOM Output (optional)
    # ------------------------------------------------------------------
    t_dicom = 0.0
    mem_dicom = 0.0
    if dicom_output:
        print("\n" + "=" * 60)
        print("STAGE 5: DICOM OUTPUT")
        print("=" * 60)
        _sync_gpu()
        _gpu_peak_mb()
        t_dicom_start = time.time()

        simulator.save_dicom_output()

        _sync_gpu()
        t_dicom = time.time() - t_dicom_start
        mem_dicom = _gpu_peak_mb()
        print(f"  Done in {t_dicom:.1f}s  (GPU peak: {mem_dicom:.0f} MB)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    stages = [
        ('init',          t_init,  mem_init),
        ('bag_gen',       t_bag,   mem_bag),
        ('fwd_model',     t_fwd,   mem_fwd),
        ('decomposer',    t_dec,   mem_dec),
        ('reconstructor', t_recon, mem_recon),
        ('dicom_output',  t_dicom, mem_dicom),
    ]
    total_time = sum(s[1] for s in stages)
    peak_mem = max(s[2] for s in stages)

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    gpu_name = _gpu_name()
    gpu_total = _gpu_total_mb()

    # Scanner info
    try:
        img_dims = scanner.recon_params['image_dims']
        gantry = scanner.machine_geometry['gantry_diameter']
        px_size = scanner.machine_geometry.get('det_spacing_y', 1.0)
        scanner_desc = (f"parallel {img_dims[0]}x{img_dims[1]}x{img_dims[2]}"
                        f" @ {px_size} mm/voxel")
    except Exception:
        scanner_desc = 'unknown'

    # Print table
    print("\n")
    print("=" * 62)
    print(f"{'Stage':<18} {'Time (s)':>10} {'%':>8} {'GPU peak (MB)':>14}")
    print("-" * 62)
    for name, t, m in stages:
        pct = (t / total_time * 100) if total_time > 0 else 0
        print(f"{name:<18} {t:>10.1f} {pct:>7.1f}% {m:>14.0f}")
    print("-" * 62)
    print(f"{'TOTAL':<18} {total_time:>10.1f} {'100.0%':>8} {peak_mem:>14.0f}")
    print("=" * 62)
    print(f"Config:  {config_name}")
    print(f"Scanner: {scanner_desc}")
    print(f"GPU:     {gpu_name} ({gpu_total:.0f} MB)")
    print()

    # Save JSON
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(results_dir,
                             f'{config_name}_{timestamp}.json')

    result = dict(
        config=config_name,
        config_path=os.path.abspath(config_path),
        timestamp=timestamp,
        scanner=scanner_desc,
        gpu=gpu_name,
        gpu_total_mb=gpu_total,
        stages={name: dict(time_s=round(t, 2),
                           pct=round(t / total_time * 100, 1)
                           if total_time > 0 else 0,
                           gpu_peak_mb=round(m, 0))
                for name, t, m in stages},
        total_time_s=round(total_time, 2),
        total_gpu_peak_mb=round(peak_mem, 0),
    )

    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Cleanup
    if simulator.logger.hasHandlers():
        simulator.logger.handlers.clear()
    del simulator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the DEBISim2 pipeline per stage')

    parser.add_argument('--config', required=True,
                        help='Path to config file')
    parser.add_argument('--sim_dir',
                        default=os.path.join(PROJECT_ROOT,
                                             'results', 'benchmark_run'),
                        help='Output directory for benchmark run')

    args = parser.parse_args()
    run_benchmark(args.config, args.sim_dir)
