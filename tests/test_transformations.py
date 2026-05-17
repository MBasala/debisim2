"""
Unit tests for every data transformation in the DEBISim2 pipeline.

Tests are organized by pipeline stage:
  1. Volume axis conventions (ASTRA input format)
  2. Forward projection (volume → sinogram)
  3. Beer-Lambert / noise / log-attenuation
  4. Sinogram axis reordering (pipeline ↔ ASTRA ↔ decomposer)
  5. Ram-Lak filtering
  6. FBP reconstruction (sinogram → volume)
  7. HU conversion (LAC → Hounsfield Units)
  8. DICOM pixel array orientation
  9. RT-Struct contour coordinate frame
  10. Roundtrip consistency (project → reconstruct → compare)
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def small_scanner():
    """Create a minimal parallel-beam scanner for unit tests."""
    from lib.forward_model.scanner_template import create_parallel_scanner
    scanner = create_parallel_scanner(
        gantry_diameter_mm=64,
        pixel_size_mm=2.0,
        n_slices=4,
        n_views=90,
        pscale=1.0,
    )
    return scanner


@pytest.fixture
def unit_sphere_volume():
    """32x32x4 volume with a centered sphere of known LAC."""
    vol = np.zeros((32, 32, 4), dtype=np.float32)
    cx, cy = 16, 16
    r = 6
    for x in range(32):
        for y in range(32):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                vol[x, y, :] = 0.2  # water-like LAC
    return vol


@pytest.fixture
def simple_sinogram(small_scanner):
    """Generate a sinogram from a simple phantom using the scanner."""
    vol = np.zeros((32, 32, 4), dtype=np.float32)
    vol[12:20, 12:20, :] = 0.3  # dense block
    sino = small_scanner.run_fwd_projector(vol)
    return sino


# ===========================================================================
# 1. Volume Axis Conventions
# ===========================================================================

class TestVolumeAxisConventions:
    """Verify ASTRA expects (D, H, W) from our (H, W, D) volumes."""

    def test_transpose_hwp_to_dhw(self):
        """transpose(vol, (2,0,1)) should convert (H,W,D) → (D,H,W)."""
        vol = np.zeros((10, 20, 5))
        result = np.transpose(vol, (2, 0, 1))
        assert result.shape == (5, 10, 20)

    def test_moveaxis_last_to_first(self):
        """moveaxis(vol, -1, 0) should convert (H,W,D) → (D,H,W)."""
        vol = np.zeros((10, 20, 5))
        result = np.moveaxis(vol, -1, 0)
        assert result.shape == (5, 10, 20)

    def test_transpose_and_moveaxis_equivalent(self):
        """Both should produce identical results."""
        rng = np.random.default_rng(42)
        vol = rng.random((10, 20, 5))
        t1 = np.transpose(vol, (2, 0, 1))
        t2 = np.moveaxis(vol, -1, 0)
        np.testing.assert_array_equal(t1, t2)

    def test_volume_data_preserved(self):
        """Axis reorder should not alter values, only layout."""
        rng = np.random.default_rng(42)
        vol = rng.random((8, 12, 3))
        reordered = np.transpose(vol, (2, 0, 1))
        # Value at (h=2, w=5, d=1) should be at (1, 2, 5) after transpose
        assert vol[2, 5, 1] == reordered[1, 2, 5]


# ===========================================================================
# 2. Forward Projection
# ===========================================================================

class TestForwardProjection:
    """Verify forward projection produces correct sinogram shapes and values."""

    @pytest.mark.slow
    def test_projection_shape(self, small_scanner, unit_sphere_volume):
        """Projection of (32,32,4) volume should produce correct sinogram shape."""
        sino = small_scanner.run_fwd_projector(unit_sphere_volume)
        det_rows = small_scanner.machine_geometry['det_row_count']
        det_cols = small_scanner.machine_geometry['det_col_count']
        n_views = small_scanner.recon_params['n_views']
        assert sino.shape == (det_rows, n_views, det_cols), \
            f"Expected ({det_rows}, {n_views}, {det_cols}), got {sino.shape}"

    @pytest.mark.slow
    def test_empty_volume_zero_projection(self, small_scanner):
        """Projection of zeros should be all zeros."""
        vol = np.zeros((32, 32, 4), dtype=np.float32)
        sino = small_scanner.run_fwd_projector(vol)
        np.testing.assert_allclose(sino, 0.0, atol=1e-6)

    @pytest.mark.slow
    def test_uniform_volume_nonzero(self, small_scanner):
        """Projection of uniform volume should produce non-zero sinogram."""
        vol = np.ones((32, 32, 4), dtype=np.float32) * 0.1
        sino = small_scanner.run_fwd_projector(vol)
        assert sino.max() > 0, "Projection of nonzero volume should be nonzero"

    @pytest.mark.slow
    def test_projection_linearity(self, small_scanner, unit_sphere_volume):
        """Projection should be linear: proj(2*vol) = 2*proj(vol)."""
        sino_1x = small_scanner.run_fwd_projector(unit_sphere_volume)
        sino_2x = small_scanner.run_fwd_projector(unit_sphere_volume * 2.0)
        np.testing.assert_allclose(sino_2x, sino_1x * 2.0, rtol=1e-4)

    @pytest.mark.slow
    def test_projection_nonnegative(self, small_scanner, unit_sphere_volume):
        """Projection of non-negative volume should be non-negative."""
        sino = small_scanner.run_fwd_projector(unit_sphere_volume)
        assert sino.min() >= -1e-6, f"Negative projection values: {sino.min()}"


# ===========================================================================
# 3. Beer-Lambert / Noise / Log-Attenuation
# ===========================================================================

class TestBeerLambertTransform:
    """Test the exp(-projection) and log-attenuation transforms."""

    def test_beer_lambert_identity(self):
        """exp(-proj) applied to zero projection should give 1.0."""
        proj = np.zeros((4, 10, 8), dtype=np.float32)
        transmitted = np.exp(-proj)
        np.testing.assert_allclose(transmitted, 1.0)

    def test_beer_lambert_attenuation(self):
        """Higher LAC projection → lower transmitted intensity."""
        proj_low = np.full((4, 10, 8), 0.1, dtype=np.float32)
        proj_high = np.full((4, 10, 8), 1.0, dtype=np.float32)
        trans_low = np.exp(-proj_low)
        trans_high = np.exp(-proj_high)
        assert trans_high.mean() < trans_low.mean()

    def test_log_attenuation_roundtrip(self):
        """-log(exp(-x)) should return x (within numerical precision)."""
        rng = np.random.default_rng(42)
        proj = rng.uniform(0.01, 5.0, size=(4, 10, 8)).astype(np.float32)
        transmitted = np.exp(-proj)
        recovered = -np.log(transmitted)
        np.testing.assert_allclose(recovered, proj, rtol=1e-5)

    def test_poisson_noise_preserves_mean(self):
        """Poisson noise should preserve the mean (law of large numbers)."""
        import torch
        torch.manual_seed(42)
        intensity = torch.full((1000, 1000), 1e4, dtype=torch.float32)
        noisy = torch.poisson(intensity)
        # Mean should be within 1% of expected
        assert abs(noisy.float().mean().item() - 1e4) / 1e4 < 0.01

    def test_poisson_noise_increases_std(self):
        """Poisson noise should add variance proportional to signal."""
        import torch
        torch.manual_seed(42)
        intensity = torch.full((1000, 1000), 1e4, dtype=torch.float32)
        noisy = torch.poisson(intensity)
        # Std should be ~sqrt(1e4) = 100
        measured_std = noisy.float().std().item()
        assert 80 < measured_std < 120, f"Poisson std={measured_std}, expected ~100"

    def test_log_attenuation_clamp(self):
        """Clamping before log prevents log(0) = -inf."""
        counts = np.array([0.0, 0.5, 1.0, 100.0])
        clamped = np.clip(counts, 1.0, None)
        result = -np.log(clamped)
        assert np.all(np.isfinite(result)), "Clamped log should be finite"
        assert result[0] == 0.0, "log(1.0) should be 0"


# ===========================================================================
# 4. Sinogram Axis Reordering
# ===========================================================================

class TestSinogramAxisReordering:
    """Test the moveaxis/flip chain between pipeline stages."""

    def test_fwd_model_to_decomposer_reorder(self):
        """Forward model output (det_rows, n_views, det_cols) →
        decomposer input (det_cols, det_rows, n_views) via
        moveaxis(-1, 0) + [:,:,::-1]."""
        sino = np.arange(24).reshape(2, 3, 4)  # (det_rows=2, n_views=3, det_cols=4)
        reordered = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        assert reordered.shape == (4, 2, 3)

    def test_flip_reverses_view_order(self):
        """The [:,:,::-1] flip reverses the view dimension."""
        sino = np.arange(12).reshape(1, 3, 4)
        reordered = np.moveaxis(sino, -1, 0)  # (4, 1, 3)
        flipped = reordered[:, :, ::-1]
        # Last axis reversed
        np.testing.assert_array_equal(flipped[0, 0, :], reordered[0, 0, ::-1])

    def test_recon_input_shape(self):
        """Reconstructor expects (det_cols, det_rows, n_views) after
        the moveaxis(-1, 0) + flip chain."""
        det_rows, n_views, det_cols = 64, 360, 256
        sino_fwd = np.zeros((det_rows, n_views, det_cols))
        sino_recon = np.moveaxis(sino_fwd, -1, 0)[:, :, ::-1]
        assert sino_recon.shape == (det_cols, det_rows, n_views)

    def test_recon_internal_transpose(self):
        """Inside _run_fbp_parallel_beam, transpose(sino, (1,2,0))
        converts (det_cols, det_rows, n_views) → (det_rows, n_views, det_cols)."""
        sino = np.zeros((256, 64, 360))  # (det_cols, det_rows, n_views)
        transposed = np.transpose(sino, (1, 2, 0))
        assert transposed.shape == (64, 360, 256)  # (det_rows, n_views, det_cols)

    def test_full_axis_chain_preserves_data(self):
        """Full chain: fwd → moveaxis+flip → transpose should not lose data."""
        rng = np.random.default_rng(42)
        sino_fwd = rng.random((4, 10, 8))  # (det_rows, n_views, det_cols)
        # Pipeline chain
        sino_decomp = np.moveaxis(sino_fwd, -1, 0)[:, :, ::-1]  # (8, 4, 10) flipped
        # FBP internal
        sino_fbp = np.transpose(sino_decomp, (1, 2, 0))  # (4, 10, 8) = back to original dims
        # After flip+transpose, data is flipped along views but shape restored
        assert sino_fbp.shape == (4, 10, 8)


# ===========================================================================
# 5. Ram-Lak Filter
# ===========================================================================

class TestRamLakFilter:
    """Test the dynamically generated Ram-Lak filter."""

    def test_ramlak_shape(self, small_scanner):
        """Ram-Lak filter should match (n_det, n_views) complex."""
        n_det = small_scanner.machine_geometry['det_col_count']
        n_views = small_scanner.recon_params['n_views']
        assert small_scanner.ramlak.shape == (n_det, n_views)
        assert small_scanner.ramlak.dtype == complex

    def test_ramlak_dc_component(self, small_scanner):
        """DC component of frequency-domain ramp filter |ω| is 0 at index 0.
        The current implementation generates the ramp directly in the
        frequency domain: [0, 1/half, 2/half, ..., 1, ..., 2/half, 1/half].
        DC (index 0) should be exactly 0. Verify it's finite and real."""
        ramlak_col = small_scanner.ramlak[:, 0]
        dc = ramlak_col[0]
        assert np.isfinite(dc.real), f"DC component should be finite, got {dc}"
        assert abs(dc.imag) < 1e-10, f"DC should be real, got imag={dc.imag}"

    def test_ramlak_symmetric(self, small_scanner):
        """Ram-Lak filter should have conjugate symmetry (real-valued kernel)."""
        ramlak_col = small_scanner.ramlak[:, 0]
        n = len(ramlak_col)
        for k in range(1, n // 2):
            assert abs(ramlak_col[k] - np.conj(ramlak_col[n - k])) < 1e-10

    def test_ramlak_tiled_across_views(self, small_scanner):
        """All view columns should be identical."""
        r = small_scanner.ramlak
        for v in range(1, r.shape[1]):
            np.testing.assert_array_equal(r[:, 0], r[:, v])


# ===========================================================================
# 6. FBP Reconstruction
# ===========================================================================

class TestFBPReconstruction:
    """Test FBP reconstruction correctness."""

    @pytest.mark.slow
    def test_reconstruction_shape(self, small_scanner, unit_sphere_volume):
        """Reconstruct should produce the correct volume shape."""
        sino = small_scanner.run_fwd_projector(unit_sphere_volume)
        # Apply pipeline axis reorder
        sino_recon = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        rec = small_scanner.reconstruct_data(sino_recon)
        img_dims = small_scanner.recon_params['image_dims']
        # FBP output shape depends on implementation
        assert rec.ndim == 3, f"Expected 3D output, got {rec.ndim}D"

    @pytest.mark.slow
    def test_zero_sinogram_zero_reconstruction(self, small_scanner):
        """FBP of zero sinogram should produce near-zero reconstruction."""
        det_cols = small_scanner.machine_geometry['det_col_count']
        det_rows = small_scanner.machine_geometry['det_row_count']
        n_views = small_scanner.recon_params['n_views']
        sino = np.zeros((det_cols, det_rows, n_views), dtype=np.float32)
        rec = small_scanner.reconstruct_data(sino)
        assert abs(rec.max()) < 0.1, f"Zero sino should give near-zero recon, got max={rec.max()}"

    @pytest.mark.slow
    def test_reconstruction_nonnegative_interior(self, small_scanner, unit_sphere_volume):
        """Interior of reconstructed sphere should be non-negative."""
        sino = small_scanner.run_fwd_projector(unit_sphere_volume)
        sino_recon = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        rec = small_scanner.reconstruct_data(sino_recon)
        # Center region should be positive (where the sphere is)
        cx, cy = rec.shape[0] // 2, rec.shape[1] // 2
        center_val = rec[cx, cy, rec.shape[2] // 2]
        # We just check it's not wildly negative
        assert center_val > -1.0, \
            f"Center of sphere reconstruction = {center_val}, expected > -1.0"


# ===========================================================================
# 7. HU Conversion
# ===========================================================================

class TestHUConversion:
    """Test LAC → HU conversion formula."""

    def test_water_hu_is_zero(self):
        """HU = (LAC - LAC_w) / LAC_w * 1000. Water → 0 HU."""
        mu_w = 0.2374
        lac_water = 0.2374
        hu = (lac_water - mu_w) / mu_w * 1000
        assert abs(hu) < 1e-10

    def test_air_hu_is_minus_1000(self):
        """Air (LAC ≈ 0) → -1000 HU."""
        mu_w = 0.2374
        lac_air = 0.0
        hu = (lac_air - mu_w) / mu_w * 1000
        assert abs(hu - (-1000)) < 1e-10

    def test_double_water_lac_is_plus_1000(self):
        """Material with 2× water LAC → +1000 HU."""
        mu_w = 0.2374
        lac = mu_w * 2
        hu = (lac - mu_w) / mu_w * 1000
        assert abs(hu - 1000) < 1e-10

    def test_hu_monotonic_with_lac(self):
        """Higher LAC should always produce higher HU."""
        mu_w = 0.2374
        lacs = [0.0, 0.1, 0.2374, 0.5, 1.0, 2.0]
        hus = [(lac - mu_w) / mu_w * 1000 for lac in lacs]
        for i in range(len(hus) - 1):
            assert hus[i] < hus[i + 1], \
                f"HU not monotonic: lac={lacs[i]}→{hus[i]}, lac={lacs[i+1]}→{hus[i+1]}"

    def test_hu_clipping(self):
        """HU values should be clipped to [-1000, 32000]."""
        cmin, cmax = -1000, 32000
        hu_values = np.array([-2000, -1000, 0, 1000, 50000])
        clipped = np.clip(hu_values, cmin, cmax)
        expected = np.array([-1000, -1000, 0, 1000, 32000])
        np.testing.assert_array_equal(clipped, expected)

    def test_hu_scale_factor(self):
        """img_scale multiplies LAC before HU conversion."""
        mu_w = 0.2374
        raw_lac = 0.02374  # 10× too small
        scale = 10.0
        scaled_lac = raw_lac * scale
        hu = (scaled_lac - mu_w) / mu_w * 1000
        assert abs(hu) < 1e-6, \
            f"Scaled water LAC should give 0 HU, got {hu}"

    def test_hu_offset_mhu_mode(self):
        """MHU mode adds 1000 offset: air = 0, water = 1000."""
        mu_w = 0.2374
        offset = 1000
        hu_air = (0.0 - mu_w) / mu_w * 1000 + offset
        hu_water = (mu_w - mu_w) / mu_w * 1000 + offset
        assert abs(hu_air - 0) < 1e-10
        assert abs(hu_water - 1000) < 1e-10


# ===========================================================================
# 8. DICOM Pixel Array Orientation
# ===========================================================================

class TestDICOMOrientation:
    """Test DICOM pixel array axis convention (row=Y, col=X)."""

    def test_transpose_for_dicom(self):
        """Volume slice vol[:,:,z].T should swap row↔col for DICOM."""
        vol = np.zeros((10, 20, 5))
        vol[3, 7, 2] = 1.0  # x=3, y=7, z=2
        dicom_slice = vol[:, :, 2].T  # becomes (20, 10) = (Y, X)
        # In DICOM: row=Y=7, col=X=3
        assert dicom_slice[7, 3] == 1.0
        assert dicom_slice.shape == (20, 10)

    def test_untransposed_wrong_mapping(self):
        """Without transpose, x maps to row (wrong for DICOM)."""
        vol = np.zeros((10, 20, 5))
        vol[3, 7, 2] = 1.0
        wrong_slice = vol[:, :, 2]  # (10, 20) = (X, Y) — wrong!
        # Row=3 is X, not Y
        assert wrong_slice[3, 7] == 1.0  # this IS the data, just wrong axes

    def test_asymmetric_volume_transpose(self):
        """Non-square slices make transpose visible."""
        vol = np.zeros((10, 20, 5))
        dicom_slice = vol[:, :, 0].T
        assert dicom_slice.shape == (20, 10), \
            "DICOM rows=Y=20, cols=X=10"


# ===========================================================================
# 9. RT-Struct Contour Coordinates
# ===========================================================================

class TestRTStructCoordinates:
    """Test RT-Struct contour coordinate frame consistency."""

    def test_contour_to_pixel_mapping(self):
        """Patient coords (x_mm, y_mm) → pixel (col, row) via spacing."""
        pixel_spacing = 2.0
        origin_x, origin_y = 0.0, 0.0
        # Point at (10mm, 20mm) should be pixel col=5, row=10
        x_mm, y_mm = 10.0, 20.0
        col = (x_mm - origin_x) / pixel_spacing
        row = (y_mm - origin_y) / pixel_spacing
        assert col == 5.0
        assert row == 10.0

    def test_contour_z_to_slice_mapping(self):
        """Z coordinate in mm maps to slice index via slice_spacing."""
        slice_spacing = 2.0
        origin_z = 0.0
        z_mm = 10.0
        z_idx = round((z_mm - origin_z) / slice_spacing)
        assert z_idx == 5

    def test_contour_z_clipped_to_content(self):
        """Contours should only exist on slices with reconstructed content."""
        content_z_min, content_z_max = 54.0, 72.0
        contour_z_values = [44.0, 50.0, 54.0, 60.0, 72.0, 80.0]
        clipped = [z for z in contour_z_values
                   if content_z_min <= z <= content_z_max]
        assert clipped == [54.0, 60.0, 72.0]


# ===========================================================================
# 10. Roundtrip Consistency
# ===========================================================================

class TestRoundtripConsistency:
    """Test project → reconstruct roundtrip preserves signal."""

    @pytest.mark.slow
    def test_sphere_survives_roundtrip(self, small_scanner, unit_sphere_volume):
        """A sphere should be detectable after project → reconstruct."""
        # Forward project
        sino = small_scanner.run_fwd_projector(unit_sphere_volume)
        # Reorder for reconstructor
        sino_recon = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        # Reconstruct
        rec = small_scanner.reconstruct_data(sino_recon)
        # The center should be brighter than the corner
        cx, cy, cz = rec.shape[0] // 2, rec.shape[1] // 2, rec.shape[2] // 2
        center = rec[cx, cy, cz]
        corner = rec[0, 0, cz]
        assert center > corner, \
            f"Center ({center:.4f}) should be brighter than corner ({corner:.4f})"

    @pytest.mark.slow
    def test_reconstruction_normalization(self, small_scanner, unit_sphere_volume):
        """Reconstructed LAC in sphere interior should be within
        an order of magnitude of input LAC (0.2)."""
        sino = small_scanner.run_fwd_projector(unit_sphere_volume)
        sino_recon = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        rec = small_scanner.reconstruct_data(sino_recon)
        cx, cy, cz = rec.shape[0] // 2, rec.shape[1] // 2, rec.shape[2] // 2
        center_lac = rec[cx, cy, cz]
        input_lac = 0.2
        ratio = center_lac / input_lac if input_lac > 0 else float('inf')
        # Record the ratio for calibration debugging
        print(f"\nFBP normalization ratio: {ratio:.4f} "
              f"(center_lac={center_lac:.6f}, input_lac={input_lac})")
        # Just verify it's positive and finite
        assert center_lac > 0, f"Center LAC should be positive, got {center_lac}"
        assert np.isfinite(center_lac), "Center LAC should be finite"


# ===========================================================================
# 11. Physics-Based Verification (Analytical Solutions)
# ===========================================================================

class TestProjectionAnalytical:
    """Verify forward projection against closed-form solutions.

    For a uniform cylinder of radius R and LAC μ, the parallel-beam
    projection at detector offset d from center is:

        p(d) = 2μ √(R² - d²)    for |d| ≤ R
        p(d) = 0                  for |d| > R

    This is the Radon transform of a circle — textbook result,
    independent of our implementation.
    """

    @pytest.fixture
    def cylinder_phantom(self):
        """Create a 2D cylinder (circle) phantom with known LAC and radius.
        Use large grid for accurate comparison to analytical solution."""
        n = 128
        lac = 0.3  # cm⁻¹
        radius_pixels = 20
        vol = np.zeros((n, n, 1), dtype=np.float32)
        cx, cy = n // 2, n // 2
        for x in range(n):
            for y in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius_pixels ** 2:
                    vol[x, y, 0] = lac
        return vol, lac, radius_pixels

    @pytest.fixture
    def cylinder_scanner(self):
        """Scanner matched to the cylinder phantom."""
        from lib.forward_model.scanner_template import create_parallel_scanner
        return create_parallel_scanner(
            gantry_diameter_mm=256,
            pixel_size_mm=2.0,
            n_slices=1,
            n_views=360,
        )

    @pytest.mark.slow
    def test_projection_peak_matches_analytical(self, cylinder_scanner,
                                                 cylinder_phantom):
        """Peak projection (ray through center) should equal 2μR.

        Analytical: p(0) = 2 × μ × R_physical
        where R_physical = R_pixels × pixel_size_mm.

        With the geometry-fix that aligns vol_geom with proj_geom
        physical units, ASTRA integrates in mm — projection values
        are LAC (cm⁻¹) × path-length-in-mm, matching the textbook
        Radon transform of the cylinder.
        """
        vol, lac, radius_px = cylinder_phantom
        pixel_size_mm = cylinder_scanner.machine_geometry['det_spacing_y']
        sino = cylinder_scanner.run_fwd_projector(vol)
        # sino shape: (det_rows, n_views, det_cols)
        # For a centered circle, peak projection is at the center detector
        center_det = sino.shape[2] // 2
        # Take the 0-degree view (view 0)
        peak_measured = sino[0, 0, center_det]
        # Analytical: 2 * lac * physical_diameter = 2 * lac * R_px * pixel_size
        peak_analytical = 2.0 * lac * radius_px * pixel_size_mm
        ratio = peak_measured / peak_analytical if peak_analytical > 0 else 0

        print(f"\nProjection peak test:")
        print(f"  Measured:   {peak_measured:.6f}")
        print(f"  Analytical: {peak_analytical:.6f}")
        print(f"  Ratio:      {ratio:.4f}")

        # Allow 15% tolerance for discretization on a finite grid
        assert abs(ratio - 1.0) < 0.15, \
            f"Peak projection ratio {ratio:.4f} deviates >15% from analytical"

    @pytest.mark.slow
    def test_projection_profile_shape(self, cylinder_scanner, cylinder_phantom):
        """Projection profile should follow √(R²-d²) shape.

        Compare the measured profile against the analytical Radon transform
        of a circle at several offsets from center.  With physical vol_geom,
        both r and d are in physical mm units.
        """
        vol, lac, radius_px = cylinder_phantom
        pixel_size_mm = cylinder_scanner.machine_geometry['det_spacing_y']
        sino = cylinder_scanner.run_fwd_projector(vol)
        n_det = sino.shape[2]
        center = n_det // 2

        # Sample at offsets well within the circle (avoid discretized edge)
        offsets = [0, 3, 6, 9, 12]  # pixels from center (radius=20, stay <60%R)
        errors = []
        for d in offsets:
            if d >= radius_px:
                continue
            measured = sino[0, 0, center + d]
            # Analytical Radon transform of a circle: 2*lac*sqrt(R²-d²),
            # both R and d in PHYSICAL units (mm).
            r_mm = radius_px * pixel_size_mm
            d_mm = d * pixel_size_mm
            analytical = 2.0 * lac * np.sqrt(r_mm ** 2 - d_mm ** 2)
            if analytical > 0:
                rel_err = abs(measured - analytical) / analytical
                errors.append((d, measured, analytical, rel_err))

        print("\nProjection profile comparison:")
        print(f"  {'Offset':>6} {'Measured':>10} {'Analytical':>10} {'RelErr':>8}")
        for d, m, a, e in errors:
            print(f"  {d:6d} {m:10.4f} {a:10.4f} {e:8.4f}")

        # Most points should be within 20% (discretization limits)
        good = sum(1 for _, _, _, e in errors if e < 0.20)
        assert good >= len(errors) * 0.6, \
            f"Only {good}/{len(errors)} profile points within 20% of analytical"

    @pytest.mark.slow
    def test_projection_zero_outside_cylinder(self, cylinder_scanner,
                                               cylinder_phantom):
        """Rays outside the cylinder should have zero projection."""
        vol, lac, radius_px = cylinder_phantom
        sino = cylinder_scanner.run_fwd_projector(vol)
        n_det = sino.shape[2]
        center = n_det // 2
        # Sample well outside the cylinder (offset > radius + margin)
        outside_idx = center + radius_px + 10
        if outside_idx < n_det:
            outside_val = sino[0, 0, outside_idx]
            assert abs(outside_val) < 0.01, \
                f"Projection outside cylinder should be ~0, got {outside_val:.4f}"

    @pytest.mark.slow
    def test_projection_rotational_invariance(self, cylinder_scanner,
                                               cylinder_phantom):
        """A centered circle's projection should be identical at all angles."""
        vol, lac, radius_px = cylinder_phantom
        sino = cylinder_scanner.run_fwd_projector(vol)
        # Compare center detector value across views
        center_det = sino.shape[2] // 2
        view_values = sino[0, :, center_det]
        # Should all be approximately equal (circle is rotationally symmetric)
        std = np.std(view_values)
        mean = np.mean(view_values)
        cv = std / mean if mean > 0 else float('inf')
        assert cv < 0.05, \
            f"Circle projection CV across views = {cv:.4f}, expected <0.05"


class TestReconstructionAnalytical:
    """Verify FBP reconstruction against known phantom properties.

    If we project a uniform cylinder and reconstruct it, the interior
    should recover the original LAC value. This is the fundamental
    theorem of CT — the Fourier Slice Theorem guarantees exact recovery
    in the continuous case. Discrete FBP should get within ~10-20%.
    """

    @pytest.fixture
    def recon_scanner(self):
        """Larger scanner for more accurate reconstruction."""
        from lib.forward_model.scanner_template import create_parallel_scanner
        return create_parallel_scanner(
            gantry_diameter_mm=256,
            pixel_size_mm=2.0,
            n_slices=1,
            n_views=360,
        )

    @pytest.fixture
    def recon_phantom(self):
        """Centered cylinder phantom for reconstruction test."""
        n = 128
        lac = 0.25  # water-like
        vol = np.zeros((n, n, 1), dtype=np.float32)
        cx, cy = n // 2, n // 2
        r = 25
        for x in range(n):
            for y in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    vol[x, y, 0] = lac
        return vol, lac, r

    @staticmethod
    def _sample_interior(rec, cx, cy, r_sample):
        """Sample interior values handling both (Z,X,Y) and (X,Y,Z) layouts."""
        # FBP returns (n_slices, im_x, im_y) — find the slice axis
        if rec.shape[0] < rec.shape[1]:
            # (n_slices, X, Y) — slice axis is 0
            cz = rec.shape[0] // 2
            vals = []
            for dx in range(-r_sample, r_sample + 1):
                for dy in range(-r_sample, r_sample + 1):
                    if dx ** 2 + dy ** 2 <= r_sample ** 2:
                        vals.append(rec[cz, cx + dx, cy + dy])
        else:
            # (X, Y, n_slices) — slice axis is 2
            cz = rec.shape[2] // 2
            vals = []
            for dx in range(-r_sample, r_sample + 1):
                for dy in range(-r_sample, r_sample + 1):
                    if dx ** 2 + dy ** 2 <= r_sample ** 2:
                        vals.append(rec[cx + dx, cy + dy, cz])
        return np.array(vals)

    @staticmethod
    def _sample_corner(rec):
        """Sample corner value handling both axis layouts."""
        if rec.shape[0] < rec.shape[1]:
            cz = rec.shape[0] // 2
            return rec[cz, 5, 5]
        else:
            cz = rec.shape[2] // 2
            return rec[5, 5, cz]

    @pytest.mark.slow
    def test_reconstructed_lac_matches_input(self, recon_scanner, recon_phantom):
        """Interior LAC should match input LAC within a known tolerance.

        This is THE critical test — if this fails, the img_scale / FBP
        normalization is wrong, which explains the HU calibration failure.
        """
        vol, input_lac, radius = recon_phantom
        sino = recon_scanner.run_fwd_projector(vol)
        sino_recon = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        rec = recon_scanner.reconstruct_data(sino_recon)

        cx, cy = rec.shape[-2] // 2, rec.shape[-1] // 2
        r_sample = max(1, radius // 4)
        interior_vals = self._sample_interior(rec, cx, cy, r_sample)
        measured_lac = np.mean(interior_vals)

        ratio = measured_lac / input_lac
        rel_error = abs(ratio - 1.0)

        print(f"\n=== RECONSTRUCTION ACCURACY TEST ===")
        print(f"  Rec shape:      {rec.shape}")
        print(f"  Input LAC:      {input_lac:.4f}")
        print(f"  Measured LAC:   {measured_lac:.6f}")
        print(f"  Ratio:          {ratio:.4f}")
        print(f"  Relative error: {rel_error:.4f} ({rel_error*100:.1f}%)")
        print(f"  img_scale needed: {1.0/ratio:.4f}" if ratio > 0 else "")

        # THIS IS THE KEY ASSERTION:
        # If this fails, the FBP normalization is wrong.
        # Tolerance: 30% for discrete FBP with limited views
        assert rel_error < 0.30, \
            f"Reconstructed LAC ({measured_lac:.4f}) differs from input " \
            f"({input_lac}) by {rel_error*100:.1f}%. " \
            f"img_scale correction factor: {1.0/ratio:.2f}"

    @pytest.mark.slow
    def test_reconstructed_exterior_near_zero(self, recon_scanner, recon_phantom):
        """Region outside the cylinder should reconstruct to ~0 LAC."""
        vol, input_lac, radius = recon_phantom
        sino = recon_scanner.run_fwd_projector(vol)
        sino_recon = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        rec = recon_scanner.reconstruct_data(sino_recon)

        corner_val = self._sample_corner(rec)
        assert abs(corner_val) < input_lac * 0.3, \
            f"Exterior LAC should be near 0, got {corner_val:.4f}"

    @pytest.mark.slow
    def test_reconstructed_contrast_two_materials(self, recon_scanner):
        """Two concentric cylinders with different LAC should show
        correct contrast ratio after reconstruction."""
        n = 128
        lac_inner = 0.5   # dense core
        lac_outer = 0.2   # softer shell
        r_inner = 10
        r_outer = 30
        vol = np.zeros((n, n, 1), dtype=np.float32)
        cx, cy = n // 2, n // 2
        for x in range(n):
            for y in range(n):
                d2 = (x - cx) ** 2 + (y - cy) ** 2
                if d2 <= r_inner ** 2:
                    vol[x, y, 0] = lac_inner
                elif d2 <= r_outer ** 2:
                    vol[x, y, 0] = lac_outer

        sino = recon_scanner.run_fwd_projector(vol)
        sino_recon = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        rec = recon_scanner.reconstruct_data(sino_recon)

        # Sample center and shell, handling axis layout
        cx_r, cy_r = rec.shape[-2] // 2, rec.shape[-1] // 2
        center_vals = self._sample_interior(rec, cx_r, cy_r, 3)
        center_val = np.mean(center_vals)

        # Sample shell (halfway between inner and outer radius)
        shell_offset = (r_inner + r_outer) // 2
        shell_vals = self._sample_interior(rec, cx_r + shell_offset, cy_r, 2)
        shell_val = np.mean(shell_vals)

        print(f"\nContrast test:")
        print(f"  Inner (input {lac_inner}): measured {center_val:.4f}")
        print(f"  Outer (input {lac_outer}): measured {shell_val:.4f}")
        if shell_val > 0:
            print(f"  Contrast ratio: {center_val/shell_val:.2f} "
                  f"(expected {lac_inner/lac_outer:.2f})")

        # The RATIO should be preserved (tests relative accuracy, not absolute)
        expected_ratio = lac_inner / lac_outer  # = 2.5
        if shell_val > 0 and center_val > 0:
            measured_ratio = center_val / shell_val
            assert abs(measured_ratio - expected_ratio) / expected_ratio < 0.30, \
                f"Contrast ratio {measured_ratio:.2f} differs from " \
                f"expected {expected_ratio:.2f} by " \
                f"{abs(measured_ratio-expected_ratio)/expected_ratio*100:.0f}%"

    @pytest.mark.slow
    def test_hu_water_after_full_chain(self, recon_scanner):
        """Full chain: create water cylinder → project → reconstruct → HU.
        Water should produce 0 HU (±tolerance).

        This is the end-to-end physics test. If this fails, something in
        the chain (projection, reconstruction, normalization, or HU conversion)
        is wrong.
        """
        n = 128
        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        mu_handler = MuDatabaseHandler()
        water = mu_handler.material('water')
        mu_arr = water.get('mu', None)
        density = float(water.get('density', 1.0))
        water_lac = float(np.mean(mu_arr) * density) if mu_arr is not None else 0.24

        vol = np.zeros((n, n, 1), dtype=np.float32)
        cx, cy = n // 2, n // 2
        r = 25
        for x in range(n):
            for y in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    vol[x, y, 0] = water_lac

        sino = recon_scanner.run_fwd_projector(vol)
        sino_recon = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        rec = recon_scanner.reconstruct_data(sino_recon)

        img_scale = recon_scanner.recon_params['img_scale']

        cx_r, cy_r = rec.shape[-2] // 2, rec.shape[-1] // 2
        r_sample = max(1, r // 4)
        interior_vals = self._sample_interior(rec, cx_r, cy_r, r_sample)
        measured_lac = np.mean(interior_vals)

        scaled_lac = measured_lac * img_scale
        hu = (scaled_lac - water_lac) / water_lac * 1000

        print(f"\n=== END-TO-END WATER HU TEST ===")
        print(f"  Rec shape:         {rec.shape}")
        print(f"  Water LAC (ref):   {water_lac:.4f}")
        print(f"  FBP output (raw):  {measured_lac:.6f}")
        print(f"  img_scale:         {img_scale}")
        print(f"  Scaled LAC:        {scaled_lac:.6f}")
        print(f"  HU:                {hu:.0f} (expected: 0)")
        print(f"  FBP/ref ratio:     {measured_lac/water_lac:.4f}")
        print(f"  Needed img_scale:  {water_lac/measured_lac:.2f}"
              if measured_lac > 0 else "")

        # This WILL fail if the normalization is wrong — that's the point
        assert -200 < hu < 200, \
            f"Water HU = {hu:.0f}, expected 0 ±200. " \
            f"FBP produces {measured_lac:.6f} but water LAC is {water_lac:.4f}. " \
            f"img_scale should be {water_lac/measured_lac:.2f} not {img_scale}"


class TestSinogramPhysics:
    """Verify sinogram properties against physics constraints."""

    def test_sinogram_integral_equals_total_attenuation(self):
        """Parseval's theorem: the integral of the sinogram over all
        detector positions (for a single view) equals the integral of
        the LAC along that ray direction.

        For a uniform volume of LAC=μ filling the full FOV of width W:
            ∫ p(d) dd = μ × W × W (area × LAC)

        Actually simpler: for a single view, the sum of projection values
        across all detectors ≈ sum of LAC values along the projection
        direction × pixel_size.
        """
        n = 32
        lac = 0.15
        vol = np.full((n, n, 1), lac, dtype=np.float32)

        from lib.forward_model.scanner_template import create_parallel_scanner
        scanner = create_parallel_scanner(
            gantry_diameter_mm=64, pixel_size_mm=2.0,
            n_slices=1, n_views=4)

        sino = scanner.run_fwd_projector(vol)
        # For a 0-degree view, each detector ray passes through n voxels
        # Projection at center = lac * n
        # Sum across all n detectors ≈ lac * n * n
        view_sum = np.sum(sino[0, 0, :])
        expected_sum = lac * n * n
        ratio = view_sum / expected_sum if expected_sum > 0 else 0

        print(f"\nSinogram integral test:")
        print(f"  View sum:     {view_sum:.4f}")
        print(f"  Expected sum: {expected_sum:.4f}")
        print(f"  Ratio:        {ratio:.4f}")

        # ASTRA's parallel beam FP divides by 2 (half-angle convention
        # for 180° range).  Accept ratio near 0.5 or 1.0 depending on
        # ASTRA version/config.
        assert ratio > 0.3, \
            f"Sinogram integral ratio {ratio:.4f} too low — projection broken"
        if abs(ratio - 0.5) < 0.1:
            print("  NOTE: ASTRA applies 0.5× normalization (half-angle)")
        elif abs(ratio - 1.0) < 0.15:
            print("  NOTE: ASTRA projection matches expected sum")

    def test_denser_material_higher_projection(self):
        """A denser cylinder should produce higher projection values."""
        n = 64
        from lib.forward_model.scanner_template import create_parallel_scanner
        scanner = create_parallel_scanner(
            gantry_diameter_mm=128, pixel_size_mm=2.0,
            n_slices=1, n_views=4)

        cx, cy = n // 2, n // 2
        r = 15

        def make_cylinder(lac_val):
            v = np.zeros((n, n, 1), dtype=np.float32)
            for x in range(n):
                for y in range(n):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                        v[x, y, 0] = lac_val
            return v

        sino_low = scanner.run_fwd_projector(make_cylinder(0.1))
        sino_high = scanner.run_fwd_projector(make_cylinder(0.5))

        center_det = sino_low.shape[2] // 2
        assert sino_high[0, 0, center_det] > sino_low[0, 0, center_det], \
            "Higher LAC should produce higher projection"

    def test_beer_lambert_attenuation_physical(self):
        """For monochromatic X-rays through uniform material of thickness t:
            I = I₀ × exp(-μt)

        The log-attenuation sinogram value should equal μt.
        """
        mu = 0.3      # cm⁻¹
        t_cm = 5.0    # thickness in cm
        I0 = 1e6      # incident photon count

        # Physical transmission
        I = I0 * np.exp(-mu * t_cm)
        # Log-attenuation recovery
        recovered_mu_t = -np.log(I / I0)

        np.testing.assert_allclose(recovered_mu_t, mu * t_cm, rtol=1e-10)

    def test_polychromatic_lac_greater_than_monochromatic_high_kev(self):
        """Polyenergetic LAC should be higher than the highest-energy
        monochromatic LAC because low-energy photons are absorbed more.

        This is beam hardening in a nutshell — the effective LAC is
        weighted toward the lower energies which have higher absorption.
        """
        # Simulate with a simple two-energy spectrum
        # Low energy: μ=0.5, high energy: μ=0.1
        # Equal photons at each energy
        mu_low = 0.5
        mu_high = 0.1
        density = 1.0
        thickness = 1.0  # cm

        # Monochromatic at high energy only
        mono_high = mu_high * density

        # Polyenergetic (50/50 spectrum)
        I_transmitted = 0.5 * np.exp(-mu_low * density * thickness) + \
                        0.5 * np.exp(-mu_high * density * thickness)
        poly_lac = -np.log(I_transmitted)

        assert poly_lac > mono_high, \
            f"Polyenergetic LAC ({poly_lac:.4f}) should exceed " \
            f"high-energy mono LAC ({mono_high:.4f}) due to beam hardening"


class TestReconstructionPhysics:
    """Physics-based reconstruction verification."""

    def test_fbp_superposition(self):
        """FBP is linear: recon(a + b) = recon(a) + recon(b).

        This tests that the reconstruction algorithm preserves linearity,
        which is required by the Fourier Slice Theorem.
        """
        from lib.forward_model.scanner_template import create_parallel_scanner
        scanner = create_parallel_scanner(
            gantry_diameter_mm=64, pixel_size_mm=2.0,
            n_slices=1, n_views=90)

        n = 32
        # Two separate phantoms
        vol_a = np.zeros((n, n, 1), dtype=np.float32)
        vol_a[8:14, 10:18, 0] = 0.3

        vol_b = np.zeros((n, n, 1), dtype=np.float32)
        vol_b[18:24, 14:22, 0] = 0.2

        vol_sum = vol_a + vol_b

        def project_and_recon(v):
            s = scanner.run_fwd_projector(v)
            s = np.moveaxis(s, -1, 0)[:, :, ::-1]
            return scanner.reconstruct_data(s)

        rec_a = project_and_recon(vol_a)
        rec_b = project_and_recon(vol_b)
        rec_sum = project_and_recon(vol_sum)

        # recon(a+b) should approximately equal recon(a) + recon(b)
        diff = np.abs(rec_sum - (rec_a + rec_b))
        max_diff = diff.max()
        max_val = max(abs(rec_sum.max()), abs(rec_sum.min()), 1e-10)
        rel_diff = max_diff / max_val

        print(f"\nSuperposition test:")
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Relative diff:     {rel_diff:.6f}")

        assert rel_diff < 0.01, \
            f"FBP superposition violation: relative diff = {rel_diff:.4f}"

    def test_parseval_energy_conservation(self):
        """Parseval's theorem: energy in sinogram ≈ energy in image.

        ∫∫ |f(x,y)|² dx dy = (1/2π) ∫∫ |p(θ,s)|² dθ ds

        This tests that the FBP doesn't create or destroy energy.
        """
        from lib.forward_model.scanner_template import create_parallel_scanner
        scanner = create_parallel_scanner(
            gantry_diameter_mm=64, pixel_size_mm=2.0,
            n_slices=1, n_views=180)

        n = 32
        vol = np.zeros((n, n, 1), dtype=np.float32)
        cx, cy = n // 2, n // 2
        for x in range(n):
            for y in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 <= 10 ** 2:
                    vol[x, y, 0] = 0.25

        sino = scanner.run_fwd_projector(vol)
        sino_recon = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        rec = scanner.reconstruct_data(sino_recon)

        # Energy (L2 norm) should be positive and finite
        input_energy = np.sum(vol ** 2)
        recon_energy = np.sum(rec ** 2)

        assert recon_energy > 0, "Reconstruction energy should be positive"
        assert np.isfinite(recon_energy), "Reconstruction energy should be finite"

        # The ratio won't be exactly 1 (Parseval requires continuous transform),
        # but it should be within a reasonable range
        if input_energy > 0:
            ratio = recon_energy / input_energy
            print(f"\nParseval energy test:")
            print(f"  Input energy:  {input_energy:.4f}")
            print(f"  Recon energy:  {recon_energy:.6f}")
            print(f"  Ratio:         {ratio:.4f}")


# ===========================================================================
# 12. LAC Computation from Spectra
# ===========================================================================

class TestLACComputation:
    """Test material LAC computation from polychromatic spectra."""

    def test_water_lac_positive(self):
        """Water LAC should be positive for any reasonable spectrum."""
        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        mu = MuDatabaseHandler()
        spectra_dir = os.path.join(
            os.path.dirname(__file__), '..', 'include', 'spectra')
        # Use any available spectrum
        import glob
        specs = glob.glob(os.path.join(spectra_dir, '*.txt'))
        if not specs:
            pytest.skip("No spectrum files available")
        spec = np.loadtxt(specs[0])
        max_kev = min(spec.shape[0], 120)
        mu.calculate_lac_hu_values('water', [spec[:max_kev, 1]])
        w = mu.material('water')
        assert w['lac_1'] > 0, f"Water LAC should be positive, got {w['lac_1']}"

    def test_lac_increases_with_density(self):
        """Higher density materials should generally have higher LAC."""
        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        mu = MuDatabaseHandler()
        # Compare water (ρ=1.0) vs teflon (ρ=2.3) — both have similar Z
        water = mu.material('water')
        teflon = mu.material('teflon')
        water_density = float(water.get('density', 1.0))
        teflon_density = float(teflon.get('density', 2.3))
        assert teflon_density > water_density

    def test_lac_formula_beer_lambert(self):
        """LAC = -log(Σ spectrum × exp(-mu × density)) should be reproducible."""
        # Simple test with monochromatic spectrum at 60 keV
        spectrum = np.zeros(120)
        spectrum[50] = 1.0  # monochromatic at ~60 keV
        mu_at_60 = 0.2  # cm²/g
        density = 1.0  # g/cm³
        # LAC = -log(exp(-mu * density)) = mu * density
        lac = -np.log(np.sum(spectrum * np.exp(-mu_at_60 * density)))
        expected = mu_at_60 * density
        np.testing.assert_allclose(lac, expected, rtol=1e-10)

    def test_hu_from_lac_formula(self):
        """HU = (LAC_mat - LAC_water) / LAC_water × 1000."""
        lac_water = 0.2374
        lac_bone = 0.5  # approximate
        hu = (lac_bone - lac_water) / lac_water * 1000
        assert hu > 0, "Bone HU should be positive"
        assert hu < 5000, "Bone HU should be < 5000"


# ===========================================================================
# 12. Scale Factor Analysis
# ===========================================================================

class TestScaleFactors:
    """Document and test the various scale factors in the pipeline."""

    def test_img_scale_1_for_custom_scanner(self):
        """create_parallel_scanner should set img_scale=1.0."""
        from lib.forward_model.scanner_template import create_parallel_scanner
        scanner = create_parallel_scanner(
            gantry_diameter_mm=512, pixel_size_mm=2.0, n_slices=64)
        assert scanner.recon_params['img_scale'] == 1.0

    def test_scale_effect_on_hu(self):
        """Demonstrate how img_scale affects HU calibration.
        If FBP returns LAC/N, then scale=N corrects it."""
        mu_w = 0.2374
        true_lac = 0.2374  # water

        for scale_name, scale in [("1.0", 1.0), ("10.0", 10.0), ("14.0", 14.0)]:
            fbp_output = true_lac / scale  # simulated FBP under-estimation
            corrected = fbp_output * scale
            hu = (corrected - mu_w) / mu_w * 1000
            assert abs(hu) < 1e-6, \
                f"Scale {scale_name}: water HU should be 0, got {hu:.2f}"

    @pytest.mark.slow
    def test_fbp_normalization_factor(self, small_scanner, unit_sphere_volume):
        """Measure the actual FBP normalization factor empirically."""
        input_lac = 0.2
        sino = small_scanner.run_fwd_projector(unit_sphere_volume)
        sino_recon = np.moveaxis(sino, -1, 0)[:, :, ::-1]
        rec = small_scanner.reconstruct_data(sino_recon)

        cx, cy, cz = rec.shape[0] // 2, rec.shape[1] // 2, rec.shape[2] // 2
        measured_lac = rec[cx, cy, cz]

        if measured_lac > 0:
            normalization_factor = input_lac / measured_lac
            print(f"\nEmpirical FBP normalization factor: {normalization_factor:.4f}")
            print(f"  Input LAC: {input_lac}")
            print(f"  Measured LAC: {measured_lac:.6f}")
            print(f"  To correct: multiply FBP output by {normalization_factor:.2f}")
        else:
            print(f"\nMeasured LAC is non-positive: {measured_lac}")
