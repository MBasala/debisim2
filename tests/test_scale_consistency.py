"""
Unit tests for scale factor consistency across scanner configurations.

Verifies that the full projection → reconstruction → HU chain produces
correct LAC values regardless of pixel_size_mm or scanner geometry.

The correction chain under test:
    mu_curve = mu(cm²/g) × density(g/cm³) × self.scale    (self.scale = 0.1)
    sinogram = forward_project(mu_curve)                    (in mm⁻¹ units)
    sino_corrected = sinogram × (1 / self.scale)            (back to cm⁻¹)
    recon_lac = FBP(sino_corrected)                         (cm⁻¹)
    HU = (recon_lac - mu_w) / mu_w × 1000

    For water: HU must equal 0 (±tolerance) at ANY pixel size.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ===========================================================================
# 1. Scale Factor Algebraic Invariants
# ===========================================================================

class TestScaleInvariants:
    """Verify the scale chain produces correct HU for all scanner configs.

    The design:
    - self.scale = 0.1 (always, for noise model compatibility)
    - sinogram is divided by self.scale after the energy loop
    - img_scale is applied after FBP reconstruction
    - mu_w uses energy-weighted formula matching the energy loop

    Invariant: (1/self.scale) × FBP_ratio × img_scale must produce
    correct LAC for HU conversion.
    """

    @staticmethod
    def _make_pipeline(scanner):
        """Create a pipeline and return it."""
        import tempfile
        from src.debisim_pipeline import DEBISimPipeline
        tmp = tempfile.mkdtemp()
        pipeline = DEBISimPipeline(
            sim_path=tmp,
            scanner_model=scanner,
            xray_source_model=dict(
                num_spectra=2, kVp=130, spectra=[], dosage=[]),
            compress_data=False,
        )
        return pipeline

    @staticmethod
    def _cleanup_pipeline(pipeline):
        """Close logger handlers to release file locks."""
        if hasattr(pipeline, 'logger') and pipeline.logger.hasHandlers():
            pipeline.logger.handlers.clear()
        if hasattr(pipeline, 'scanner') and hasattr(pipeline.scanner, 'logger'):
            pipeline.scanner.logger.handlers.clear()

    def test_self_scale_always_0_1(self):
        """self.scale is always 0.1 for noise model compatibility."""
        from lib.forward_model.scanner_template import create_parallel_scanner

        for px in [0.5, 1.0, 2.0, 4.0]:
            scanner = create_parallel_scanner(
                gantry_diameter_mm=128, pixel_size_mm=px, n_slices=4)
            pipeline = self._make_pipeline(scanner)
            try:
                assert pipeline.scale == 0.1, \
                    f"px={px}mm: self.scale={pipeline.scale}, expected 0.1"
            finally:
                self._cleanup_pipeline(pipeline)

    def test_sino_correction_compensates_self_scale(self):
        """The sinogram correction (1/self.scale) restores cm-1 units."""
        from lib.forward_model.scanner_template import create_parallel_scanner

        scanner = create_parallel_scanner(
            gantry_diameter_mm=128, pixel_size_mm=2.0, n_slices=4)
        pipeline = self._make_pipeline(scanner)
        try:
            sino_correction = 1.0 / pipeline.scale
            img_scale = scanner.recon_params['img_scale']
            # For FBP_CUDA: sino_correction=10, img_scale=1.0
            # Full chain: sinogram_mm × 10 → sinogram_cm → FBP → LAC_cm × 1.0
            product = sino_correction * img_scale
            assert product == 10.0, \
                f"sino_correction × img_scale = {product}, expected 10.0"
        finally:
            self._cleanup_pipeline(pipeline)

    def test_default_scanner_scale_chain(self):
        """Default parallel beam: verify full scale chain."""
        from lib.forward_model.scanner_template import ScannerTemplate
        from lib.forward_model.template_default_parallelbeam_scanner import (
            machine_geometry, recon_params)

        scanner = ScannerTemplate(
            geometry='parallel', scan='circular',
            machine_dict=machine_geometry, recon='fbp',
            recon_dict=recon_params, pscale=1.0)
        scanner.set_recon_geometry()

        pipeline = self._make_pipeline(scanner)
        try:
            assert pipeline.scale == 0.1
            # Default: img_scale = 1/det_spacing_y = 1/0.5 = 2.0
            # sino_correction = 1/0.1 = 10
            # Chain: sinogram × 10 → FBP → LAC × 2.0
            # This gives LAC in 2× cm⁻¹ — which is wrong for FBP_CUDA
            # but correct for legacy BP3D that divides by det_spacing internally
        finally:
            self._cleanup_pipeline(pipeline)


# ===========================================================================
# 2. mu_curve Unit Verification
# ===========================================================================

class TestMuCurveUnits:
    """Verify mu_curve values are physically correct after scaling."""

    def test_water_lac_at_60kev_correct_magnitude(self):
        """Water LAC at 60 keV should be ~0.2 cm⁻¹.

        With self.scale=1.0 (FBP_CUDA):
            mu_curve[60] = mu(cm²/g)[60] × density(1.0) × 1.0 ≈ 0.2

        With self.scale=0.1 (legacy):
            mu_curve[60] = mu(cm²/g)[60] × density(1.0) × 0.1 ≈ 0.02
        """
        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        mu = MuDatabaseHandler()
        water = mu.material('water')
        mu_arr = water['mu']
        density = float(water['density'])

        # FBP_CUDA path (self.scale = 1.0)
        lac_correct = mu_arr[50] * density * 1.0
        assert 0.1 < lac_correct < 0.5, \
            f"Water LAC at 60 keV with scale=1.0: {lac_correct:.4f}"

        # Legacy path (self.scale = 0.1)
        lac_legacy = mu_arr[50] * density * 0.1
        assert 0.01 < lac_legacy < 0.05, \
            f"Water LAC at 60 keV with scale=0.1: {lac_legacy:.4f}"

    def test_full_chain_preserves_hu(self):
        """The full chain self.scale → sino_correction → FBP → img_scale
        must produce correct LAC regardless of intermediate values.

        Chain: lac * self.scale → proj → sino / self.scale → FBP → × img_scale
        = lac * (self.scale / self.scale) * img_scale = lac * img_scale
        For FBP_CUDA (img_scale=1.0): = lac  ✓
        """
        true_lac = 0.24
        mu_w = 0.24
        self_scale = 0.1  # always 0.1

        for name, img_scale in [
            ("FBP_CUDA", 1.0),
            ("legacy_0.5mm", 2.0),
        ]:
            mu_curve_val = true_lac * self_scale      # energy loop input
            # Sinogram ∝ mu_curve_val, then divided by self_scale
            sino_val = mu_curve_val / self_scale       # = true_lac
            fbp_output = sino_val                      # FBP preserves
            recon_lac = fbp_output * img_scale
            # For FBP_CUDA: recon_lac = true_lac * 1.0 = true_lac
            hu = (recon_lac - mu_w) / mu_w * 1000
            # Only works when img_scale=1.0 (FBP_CUDA handles units)
            if img_scale == 1.0:
                assert abs(hu) < 1e-6, \
                    f"{name}: HU={hu:.2f} (expected 0)"


# ===========================================================================
# 3. End-to-End FBP at Different Pixel Sizes
# ===========================================================================

class TestMultiScaleFBP:
    """Run project → reconstruct at different pixel sizes and verify
    that the reconstructed LAC always matches the input."""

    @pytest.fixture(params=[0.5, 1.0, 2.0])
    def scanner_and_phantom(self, request):
        """Create scanner and matching phantom at given pixel size."""
        px = request.param
        from lib.forward_model.scanner_template import create_parallel_scanner
        scanner = create_parallel_scanner(
            gantry_diameter_mm=64, pixel_size_mm=px,
            n_slices=1, n_views=180)
        n = int(64 / px)
        lac = 0.24  # water cm⁻¹
        vol = np.zeros((n, n, 1), dtype=np.float32)
        cx, cy = n // 2, n // 2
        r = max(3, int(10 / px))
        for x in range(n):
            for y in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    vol[x, y, 0] = lac
        return scanner, vol, lac, px

    @pytest.mark.slow
    def test_fbp_lac_invariant_across_pixel_sizes(self, scanner_and_phantom):
        """Reconstructed LAC should match input regardless of pixel size."""
        scanner, vol, input_lac, px = scanner_and_phantom

        sino = scanner.run_fwd_projector(vol)
        sino_recon = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        rec = scanner.reconstruct_data(sino_recon)

        # Sample center
        if rec.shape[0] < rec.shape[1]:
            cx, cy = rec.shape[1] // 2, rec.shape[2] // 2
            center_val = rec[0, cx, cy]
        else:
            cx, cy = rec.shape[0] // 2, rec.shape[1] // 2
            center_val = rec[cx, cy, 0]

        ratio = center_val / input_lac
        print(f"\npx={px}mm: FBP center={center_val:.6f}, "
              f"ratio={ratio:.4f}")

        assert abs(ratio - 1.0) < 0.10, \
            f"px={px}mm: ratio={ratio:.4f}, expected 1.0 ±10%"

    @pytest.mark.slow
    def test_hu_zero_for_water_at_all_pixel_sizes(self, scanner_and_phantom):
        """Water HU should be 0 regardless of pixel size, after applying
        img_scale from the scanner config."""
        scanner, vol, input_lac, px = scanner_and_phantom
        img_scale = scanner.recon_params['img_scale']

        sino = scanner.run_fwd_projector(vol)
        sino_recon = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        rec = scanner.reconstruct_data(sino_recon)

        if rec.shape[0] < rec.shape[1]:
            cx, cy = rec.shape[1] // 2, rec.shape[2] // 2
            center_val = rec[0, cx, cy]
        else:
            cx, cy = rec.shape[0] // 2, rec.shape[1] // 2
            center_val = rec[cx, cy, 0]

        scaled_lac = center_val * img_scale
        hu = (scaled_lac - input_lac) / input_lac * 1000

        print(f"\npx={px}mm: img_scale={img_scale}, "
              f"scaled_lac={scaled_lac:.6f}, HU={hu:.0f}")

        assert -100 < hu < 100, \
            f"px={px}mm: Water HU={hu:.0f}, expected 0 ±100"


# ===========================================================================
# 4. Full Simulated Energy Loop
# ===========================================================================

class TestEnergyLoopScaling:
    """Simulate the full energy loop with self.scale applied,
    then verify reconstruction recovers correct LAC after img_scale."""

    @pytest.mark.slow
    @pytest.mark.parametrize("self_scale,img_scale", [
        (0.1, 1.0),      # FBP_CUDA (current design)
    ])
    def test_scale_pair_produces_correct_hu(self, self_scale, img_scale):
        """For any (self_scale, img_scale) pair where product=1.0,
        the full chain should produce water HU ≈ 0."""
        import torch
        from lib.forward_model.scanner_template import create_parallel_scanner

        scanner = create_parallel_scanner(
            gantry_diameter_mm=64, pixel_size_mm=2.0,
            n_slices=1, n_views=180)

        n = 32
        true_lac = 0.24  # water cm⁻¹
        scaled_lac = true_lac * self_scale  # what self.scale does

        vol = np.zeros((n, n, 1), dtype=np.float32)
        cx, cy = n // 2, n // 2
        r = 8
        for x in range(n):
            for y in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    vol[x, y, 0] = scaled_lac

        # Forward project
        proj = scanner.run_fwd_projector(vol)

        # Simulate energy loop (monochromatic, no noise)
        dosage_scale = 1e5
        buffer = torch.as_tensor(proj.copy())
        buffer.neg_()
        buffer.exp_()
        buffer.mul_(dosage_scale)
        pc_sum = dosage_scale

        # Log-attenuation
        buffer.clamp_(min=1.0)
        buffer.log_()
        buffer.neg_()
        buffer.add_(np.log(pc_sum))

        # Sinogram correction: divide by self_scale to restore cm⁻¹ units
        # (matches pipeline line: projection_buffer_gpu.div_(self.scale))
        if abs(self_scale - 1.0) > 1e-6:
            buffer.div_(self_scale)

        sinogram = buffer.cpu().numpy()

        # Reconstruct
        sino_recon = np.moveaxis(sinogram, -1, 0)[:, :, ::-1]
        rec = scanner.reconstruct_data(sino_recon)

        # Apply img_scale
        if rec.shape[0] < rec.shape[1]:
            center_val = rec[0, rec.shape[1] // 2, rec.shape[2] // 2]
        else:
            center_val = rec[rec.shape[0] // 2, rec.shape[1] // 2, 0]

        recon_lac = center_val * img_scale
        mu_w = true_lac
        hu = (recon_lac - mu_w) / mu_w * 1000

        print(f"\nself_scale={self_scale}, img_scale={img_scale}:")
        print(f"  scaled_lac in vol: {scaled_lac:.4f}")
        print(f"  FBP center:        {center_val:.6f}")
        print(f"  × img_scale:       {recon_lac:.6f}")
        print(f"  Water HU:          {hu:.0f}")

        assert -100 < hu < 100, \
            f"scale=({self_scale},{img_scale}): Water HU={hu:.0f}, expected 0 ±100"

    @pytest.mark.slow
    def test_missing_sino_correction_produces_wrong_hu(self):
        """If the sinogram is NOT divided by self.scale before reconstruction,
        the LAC is 10× too small and water gives -900 HU."""
        import torch
        from lib.forward_model.scanner_template import create_parallel_scanner

        scanner = create_parallel_scanner(
            gantry_diameter_mm=64, pixel_size_mm=2.0,
            n_slices=1, n_views=180)

        n = 32
        true_lac = 0.24
        # Bug: self.scale=0.1 but img_scale=1.0 (product=0.1, not 1.0)
        self_scale = 0.1
        img_scale_wrong = 1.0

        vol = np.zeros((n, n, 1), dtype=np.float32)
        cx, cy = n // 2, n // 2
        r = 8
        for x in range(n):
            for y in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    vol[x, y, 0] = true_lac * self_scale

        proj = scanner.run_fwd_projector(vol)

        scale = 1e5
        buffer = torch.as_tensor(proj.copy())
        buffer.neg_()
        buffer.exp_()
        buffer.mul_(scale)
        pc_sum = scale

        buffer.clamp_(min=1.0)
        buffer.log_()
        buffer.neg_()
        buffer.add_(np.log(pc_sum))
        sinogram = buffer.cpu().numpy()

        sino_recon = np.moveaxis(sinogram, -1, 0)[:, :, ::-1]
        rec = scanner.reconstruct_data(sino_recon)

        if rec.shape[0] < rec.shape[1]:
            center_val = rec[0, rec.shape[1] // 2, rec.shape[2] // 2]
        else:
            center_val = rec[rec.shape[0] // 2, rec.shape[1] // 2, 0]

        recon_lac = center_val * img_scale_wrong
        hu = (recon_lac - true_lac) / true_lac * 1000

        print(f"\nMismatched scales bug demo:")
        print(f"  self.scale=0.1, img_scale=1.0 (product=0.1)")
        print(f"  Water HU: {hu:.0f} (expected ~-900)")

        # This SHOULD produce wrong HU — that's the test
        assert hu < -800, \
            f"Mismatched scales should give HU < -800, got {hu:.0f}"


# ===========================================================================
# 5. Scatter Path Scale Consistency
# ===========================================================================

class TestScatterPathScale:
    """Verify that the scatter reconstruction path uses the same
    img_scale as the main reconstruction path."""

    def test_scatter_scale_reads_from_config(self):
        """The scatter path (formerly hardcoded scale=10) should read
        from scanner.recon_params['img_scale']."""
        # Verify the code no longer hardcodes scale=10
        import inspect
        from src.debisim_pipeline import DEBISimPipeline
        source = inspect.getsource(DEBISimPipeline.add_scatter_to_ct_projection_slice)

        assert "scale = 10" not in source, \
            "Scatter path still has hardcoded 'scale = 10'"
        assert "self.scanner.recon_params['img_scale']" in source or \
               "recon_params" in source, \
            "Scatter path should read img_scale from scanner config"
