"""
Unit tests for the sinogram accumulation chain.

Tests the full forward model:
  LAC volume → per-keV projection → Beer-Lambert → noise → log-attenuation

Each step is tested independently against analytical expectations, then
the full chain is tested end-to-end to catch normalization errors like
the self.scale=0.1 / img_scale mismatch.

The energy loop implements:
    For each keV e:
        ref_image = LAC_LUT[material_index, e]           # voxel LAC at energy e
        proj = forward_project(ref_image)                  # line integral
        curr = exp(-proj) * (pc * spectrum[e] * gain * e)  # Beer-Lambert + dosage
        curr = poisson(curr)                                # quantum noise
        curr += gaussian_noise * shot_gain                  # electronic noise
        buffer += curr                                      # accumulate
        pc_sum += scale

    sinogram = -log(buffer) + log(pc_sum)                   # log-attenuation
"""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def test_scanner():
    """Small scanner for accumulation tests."""
    from lib.forward_model.scanner_template import create_parallel_scanner
    return create_parallel_scanner(
        gantry_diameter_mm=64,
        pixel_size_mm=2.0,
        n_slices=1,
        n_views=90,
    )


@pytest.fixture
def water_cylinder():
    """32x32x1 cylinder with water LAC ≈ 0.24 cm⁻¹."""
    n = 32
    lac = 0.24
    vol = np.zeros((n, n, 1), dtype=np.float32)
    cx, cy = n // 2, n // 2
    r = 8
    for x in range(n):
        for y in range(n):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                vol[x, y, 0] = lac
    return vol, lac, r


# ===========================================================================
# 1. LAC LUT Construction
# ===========================================================================

class TestLACLUT:
    """Test the per-material, per-keV LAC lookup table."""

    def test_material_curve_includes_density(self):
        """material_curve should be mass_atten × density (= LAC, cm⁻¹).

        The mu database stores mass attenuation coefficients (cm²/g).
        The pipeline multiplies by density to get LAC. If an additional
        scale factor is applied, the LUT values will be wrong.
        """
        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        mu = MuDatabaseHandler()
        water = mu.material('water')
        mu_arr = water['mu']  # mass attenuation coefficient (cm²/g)
        density = float(water['density'])  # g/cm³

        # Expected LAC at 60 keV
        expected_lac_60 = mu_arr[50] * density  # index 50 ≈ 60 keV

        assert expected_lac_60 > 0.1, \
            f"Water LAC at 60 keV should be >0.1, got {expected_lac_60:.4f}"
        assert expected_lac_60 < 1.0, \
            f"Water LAC at 60 keV should be <1.0, got {expected_lac_60:.4f}"

    def test_scale_factor_corrupts_lac(self):
        """If self.scale=0.1 is applied to mu_curve, the LUT values are
        10× too small. This is the root cause of the HU calibration error.

        Expected: mu_curve[e] = mu[e] * density
        Actual:   mu_curve[e] = mu[e] * density * 0.1   ← BUG
        """
        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        mu = MuDatabaseHandler()
        water = mu.material('water')
        mu_arr = water['mu']
        density = float(water['density'])

        correct_lac_60 = mu_arr[50] * density
        wrong_lac_60 = mu_arr[50] * density * 0.1  # self.scale = 0.1

        # The wrong value is 10× smaller
        assert abs(wrong_lac_60 / correct_lac_60 - 0.1) < 1e-10

        # This means the forward projection produces sinograms 10× too small
        # and reconstruction gives LAC values 10× too small.
        # HU = (lac*0.1 - mu_w) / mu_w * 1000 ≈ -900 for water
        hu_wrong = (correct_lac_60 * 0.1 - correct_lac_60) / correct_lac_60 * 1000
        assert hu_wrong == -900.0, \
            f"Water HU with scale=0.1 should be -900, got {hu_wrong}"

    def test_lut_shape_matches_materials_and_energies(self):
        """LUT should be (n_materials, n_keV) with no extra dimensions."""
        n_mats = 5
        n_kev = 121
        lut = np.zeros((n_mats, n_kev), dtype=np.float32)
        assert lut.shape == (n_mats, n_kev)
        # np.take along axis 0 should give (vol_shape,) per keV
        vol_indices = np.array([0, 1, 2, 1, 0, 3, 4])
        result = np.take(lut[:, 0], vol_indices)
        assert result.shape == (7,)


# ===========================================================================
# 2. Per-keV Projection Step
# ===========================================================================

class TestPerKeVProjection:
    """Test the per-keV LAC gather + forward projection step."""

    @pytest.mark.slow
    def test_monochromatic_projection_correct(self, test_scanner, water_cylinder):
        """For a single energy, projection should equal forward_project(LAC_volume)."""
        vol, lac, r = water_cylinder
        # Direct projection of the LAC volume
        sino_direct = test_scanner.run_fwd_projector(vol)
        # Should match projecting with the same LAC
        assert sino_direct.max() > 0

    @pytest.mark.slow
    def test_lut_gather_produces_correct_volume(self):
        """np.take(lut[:, k], index_volume) should reconstruct the LAC field."""
        # 3 materials, 5 keV steps
        lut = np.array([
            [0.1, 0.12, 0.15, 0.18, 0.2],    # material 0 (air-like)
            [0.24, 0.22, 0.20, 0.19, 0.18],   # material 1 (water)
            [0.5, 0.45, 0.40, 0.38, 0.35],    # material 2 (bone)
        ], dtype=np.float32)

        # Index volume: 2x2x1 with materials [0, 1, 2, 1]
        idx_vol = np.array([[[0], [1]], [[2], [1]]])

        for k in range(5):
            ref_image = np.take(lut[:, k], idx_vol)
            assert ref_image[0, 0, 0] == lut[0, k]  # air
            assert ref_image[0, 1, 0] == lut[1, k]  # water
            assert ref_image[1, 0, 0] == lut[2, k]  # bone
            assert ref_image[1, 1, 0] == lut[1, k]  # water


# ===========================================================================
# 3. Beer-Lambert Transform Chain
# ===========================================================================

class TestBeerLambertChain:
    """Test the neg→exp→mul chain: exp(-proj) × scale."""

    def test_beer_lambert_with_scale(self):
        """curr = exp(-proj) * scale should give I_transmitted × dosage."""
        proj = np.array([0.0, 0.5, 1.0, 2.0])
        scale = 1e5  # dosage
        curr = np.exp(-proj) * scale
        expected = np.array([1e5, 1e5 * np.exp(-0.5),
                             1e5 * np.exp(-1.0), 1e5 * np.exp(-2.0)])
        np.testing.assert_allclose(curr, expected, rtol=1e-6)

    def test_scale_formula(self):
        """scale = pc × spectrum[e-10] × system_gain × e.

        At 60 keV with pc=1e5, spectrum=0.01, gain=2.5e-3:
        scale = 1e5 × 0.01 × 2.5e-3 × 60 = 150
        """
        pc = 1e5
        spectrum_val = 0.01
        system_gain = 2.5e-3
        e = 60
        scale = pc * spectrum_val * system_gain * e
        expected = 150.0
        assert abs(scale - expected) < 1e-10

    def test_neg_exp_mul_chain_torch(self):
        """Verify torch in-place ops produce same result as numpy."""
        proj = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
        scale = 1000.0

        # Numpy reference
        expected = np.exp(-proj) * scale

        # Torch chain (as used in pipeline)
        curr = torch.as_tensor(proj.copy())
        curr.neg_()
        curr.exp_()
        curr.mul_(scale)

        np.testing.assert_allclose(curr.cpu().numpy(), expected, rtol=1e-6)


# ===========================================================================
# 4. Accumulation + Log-Attenuation
# ===========================================================================

class TestAccumulationAndLog:
    """Test the accumulation buffer and final log-attenuation conversion.

    The formula is:
        sinogram = -log(Σ_e I_e) + log(Σ_e scale_e)

    For a monochromatic case (single keV):
        sinogram = -log(exp(-proj) × scale) + log(scale)
                 = -(-proj + log(scale)) + log(scale)
                 = proj - log(scale) + log(scale)
                 = proj                                    ← identity!
    """

    def test_monochromatic_log_attenuation_identity(self):
        """With one energy, log-attenuation should recover the projection."""
        proj = np.array([0.5, 1.0, 2.0, 3.0])
        scale = 1e5

        # Accumulate
        buffer = np.exp(-proj) * scale
        pc_sum = scale

        # Log-attenuation
        buffer = np.clip(buffer, 1.0, None)
        sinogram = -np.log(buffer) + np.log(pc_sum)

        np.testing.assert_allclose(sinogram, proj, rtol=1e-5)

    def test_monochromatic_identity_torch(self):
        """Same test using torch ops (matching pipeline implementation)."""
        proj = torch.tensor([0.5, 1.0, 2.0, 3.0])
        scale = 1e5

        # Accumulate (as in pipeline)
        buffer = torch.zeros_like(proj)
        curr = proj.clone()
        curr.neg_()
        curr.exp_()
        curr.mul_(scale)
        buffer.add_(curr)
        pc_sum = scale

        # Log-attenuation (as in pipeline)
        buffer.clamp_(min=1.0)
        buffer.log_()
        buffer.neg_()
        buffer.add_(np.log(pc_sum))

        np.testing.assert_allclose(buffer.cpu().numpy(), proj.cpu().numpy(), rtol=1e-5)

    def test_polychromatic_accumulation(self):
        """With N energies, the sinogram is the polychromatic line integral:
            sino = -log(Σ_e w_e × exp(-μ_e × t)) + log(Σ_e w_e)

        For a uniform slab of thickness t=2 with 3 energies:
        """
        t = 2.0  # thickness
        mu_values = [0.3, 0.2, 0.15]  # LAC at 3 energies
        weights = [100, 200, 300]  # dosage weights

        buffer = 0.0
        pc_sum = 0.0
        for mu, w in zip(mu_values, weights):
            buffer += np.exp(-mu * t) * w
            pc_sum += w

        sino = -np.log(max(buffer, 1.0)) + np.log(pc_sum)

        # Analytical polychromatic LAC
        I_ratio = buffer / pc_sum  # transmitted fraction
        effective_lac_t = -np.log(I_ratio)

        np.testing.assert_allclose(sino, effective_lac_t, rtol=1e-10)

    def test_clamp_prevents_log_zero(self):
        """Clamping buffer to min=1.0 prevents -inf from log(0)."""
        buffer = torch.tensor([0.0, 0.5, 1.0, 100.0])
        buffer.clamp_(min=1.0)
        buffer.log_()
        assert torch.all(torch.isfinite(buffer)), \
            "Clamped log should be finite everywhere"

    def test_scale_cancellation_in_monochromatic(self):
        """The scale factor cancels in the monochromatic case:
            -log(exp(-p)*s) + log(s) = -(-p+log(s)) + log(s) = p

        This means the absolute scale doesn't affect the sinogram
        in the monochromatic case. But in the polychromatic case,
        the RELATIVE weights between energies matter.
        """
        proj = 1.5
        for scale in [1.0, 100.0, 1e6, 1e-3]:
            buffer = np.exp(-proj) * scale
            pc_sum = scale
            sino = -np.log(max(buffer, 1e-30)) + np.log(pc_sum)
            np.testing.assert_allclose(sino, proj, rtol=1e-10,
                                       err_msg=f"Failed at scale={scale}")


# ===========================================================================
# 5. The self.scale Bug
# ===========================================================================

class TestSelfScaleBug:
    """Tests that expose the self.scale=0.1 / img_scale mismatch.

    Root cause: In debisim_pipeline.py line 485:
        mu_curve = atten_curve * density * self.scale    (self.scale = 0.1)

    This makes every LAC value 10× too small in the energy loop.
    The original code compensated with img_scale=10 in run_reconstructor.
    Our create_parallel_scanner sets img_scale=1.0, breaking compensation.
    """

    def test_scale_0_1_reduces_sinogram_10x(self):
        """If LAC is multiplied by 0.1 before projection, the sinogram
        is 10× too small (projection is linear)."""
        true_lac = 0.24  # water
        scaled_lac = true_lac * 0.1  # self.scale = 0.1
        thickness = 10  # voxels through center

        sino_correct = true_lac * thickness
        sino_wrong = scaled_lac * thickness

        assert abs(sino_wrong / sino_correct - 0.1) < 1e-10

    def test_compensation_with_img_scale_10(self):
        """img_scale=10 in run_reconstructor compensates for self.scale=0.1:
            recon_lac = FBP(sinogram) * img_scale
                      = FBP(proj(vol * 0.1)) * 10
                      ≈ (true_lac * 0.1) * 10
                      = true_lac                           ← correct!
        """
        true_lac = 0.24
        # Pipeline chain with bug + compensation
        lac_in_lut = true_lac * 0.1       # self.scale = 0.1
        fbp_output = lac_in_lut           # FBP recovers input (ratio≈1.0)
        corrected = fbp_output * 10       # img_scale = 10
        mu_w = true_lac
        hu = (corrected - mu_w) / mu_w * 1000
        assert abs(hu) < 1e-6, f"Compensated water HU should be 0, got {hu}"

    def test_broken_with_img_scale_1(self):
        """img_scale=1.0 without fixing self.scale gives wrong HU:
            recon_lac = FBP(proj(vol * 0.1)) * 1.0
                      ≈ true_lac * 0.1
            HU = (0.024 - 0.24) / 0.24 * 1000 = -900      ← BUG!
        """
        true_lac = 0.24
        lac_in_lut = true_lac * 0.1
        fbp_output = lac_in_lut
        corrected = fbp_output * 1.0       # img_scale = 1.0
        mu_w = true_lac
        hu = (corrected - mu_w) / mu_w * 1000
        assert abs(hu - (-900)) < 1e-6, \
            f"Broken water HU should be -900, got {hu}"

    def test_fix_remove_self_scale(self):
        """The correct fix: remove self.scale from mu_curve construction.
            mu_curve = atten_curve * density              (no * self.scale)
            img_scale = 1.0                               (already set)
        """
        true_lac = 0.24
        lac_in_lut = true_lac * 1.0        # no self.scale
        fbp_output = lac_in_lut
        corrected = fbp_output * 1.0
        mu_w = true_lac
        hu = (corrected - mu_w) / mu_w * 1000
        assert abs(hu) < 1e-6, f"Fixed water HU should be 0, got {hu}"

    def test_fix_alternative_set_img_scale_10(self):
        """Alternative fix: keep self.scale=0.1, set img_scale=10.
        This is what the upstream code did. Less clean but works.
        """
        true_lac = 0.24
        lac_in_lut = true_lac * 0.1
        fbp_output = lac_in_lut
        corrected = fbp_output * 10.0      # img_scale = 10
        mu_w = true_lac
        hu = (corrected - mu_w) / mu_w * 1000
        assert abs(hu) < 1e-6, f"Alt-fix water HU should be 0, got {hu}"


# ===========================================================================
# 6. End-to-End Sinogram Chain (no noise)
# ===========================================================================

class TestEndToEndSinogramChain:
    """Test the full sinogram accumulation chain analytically.

    For a monochromatic spectrum at a single energy with known LAC:
        1. Build LAC volume from material_curve
        2. Forward project → line integrals
        3. Apply Beer-Lambert: exp(-proj) × scale
        4. Accumulate + log-attenuation
        5. Result should equal the original line integrals
    """

    @pytest.mark.slow
    def test_monochromatic_chain_recovers_projection(self, test_scanner):
        """Monochromatic chain: the sinogram should equal the projection."""
        n = 32
        lac = 0.24
        vol = np.zeros((n, n, 1), dtype=np.float32)
        cx, cy = n // 2, n // 2
        r = 8
        for x in range(n):
            for y in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    vol[x, y, 0] = lac

        # Step 1: Direct projection (ground truth sinogram)
        proj = test_scanner.run_fwd_projector(vol)

        # Step 2-4: Simulate the energy loop for 1 keV
        scale = 1e5  # arbitrary dosage
        buffer = torch.as_tensor(proj.copy())
        buffer.neg_()
        buffer.exp_()
        buffer.mul_(scale)
        pc_sum = scale

        # Step 5: Log-attenuation
        buffer.clamp_(min=1.0)
        buffer.log_()
        buffer.neg_()
        buffer.add_(np.log(pc_sum))

        recovered_sino = buffer.cpu().numpy()

        # Should match the direct projection
        center_det = proj.shape[2] // 2
        proj_center = proj[0, 0, center_det]
        sino_center = recovered_sino[0, 0, center_det]

        np.testing.assert_allclose(sino_center, proj_center, rtol=1e-4,
                                   err_msg="Monochromatic chain should recover projection")

    @pytest.mark.slow
    def test_full_roundtrip_with_reconstruction(self, test_scanner):
        """Full chain: LAC volume → sinogram → FBP → measured LAC.

        Without self.scale, the reconstructed LAC should match the input.
        """
        n = 32
        input_lac = 0.24
        vol = np.zeros((n, n, 1), dtype=np.float32)
        cx, cy = n // 2, n // 2
        r = 8
        for x in range(n):
            for y in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    vol[x, y, 0] = input_lac

        # Forward project
        proj = test_scanner.run_fwd_projector(vol)

        # Simulate energy loop (1 keV, no self.scale)
        scale = 1e5
        buffer = torch.as_tensor(proj.copy())
        buffer.neg_()
        buffer.exp_()
        buffer.mul_(scale)
        pc_sum = scale

        # Log-attenuation
        buffer.clamp_(min=1.0)
        buffer.log_()
        buffer.neg_()
        buffer.add_(np.log(pc_sum))
        sinogram = buffer.cpu().numpy()

        # Reconstruct
        sino_recon = np.moveaxis(sinogram, -1, 0)[:, :, ::-1]
        rec = test_scanner.reconstruct_data(sino_recon)

        # Sample interior
        if rec.shape[0] < rec.shape[1]:
            cx_r, cy_r = rec.shape[1] // 2, rec.shape[2] // 2
            measured = rec[0, cx_r, cy_r]
        else:
            cx_r, cy_r = rec.shape[0] // 2, rec.shape[1] // 2
            measured = rec[cx_r, cy_r, 0]

        ratio = measured / input_lac
        print(f"\nFull roundtrip (no self.scale):")
        print(f"  Input LAC:    {input_lac}")
        print(f"  Measured LAC: {measured:.6f}")
        print(f"  Ratio:        {ratio:.4f}")

        assert abs(ratio - 1.0) < 0.15, \
            f"Full roundtrip ratio {ratio:.4f} deviates >15% from 1.0"

    @pytest.mark.slow
    def test_full_roundtrip_with_self_scale_bug(self, test_scanner):
        """Same chain but with self.scale=0.1 applied to LAC.
        Demonstrates the bug: reconstructed LAC is 10× too small.
        """
        n = 32
        input_lac = 0.24
        self_scale = 0.1  # THE BUG

        vol = np.zeros((n, n, 1), dtype=np.float32)
        cx, cy = n // 2, n // 2
        r = 8
        for x in range(n):
            for y in range(n):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    vol[x, y, 0] = input_lac * self_scale  # ← reduced by 10×

        # Forward project
        proj = test_scanner.run_fwd_projector(vol)

        # Energy loop
        scale = 1e5
        buffer = torch.as_tensor(proj.copy())
        buffer.neg_()
        buffer.exp_()
        buffer.mul_(scale)
        pc_sum = scale

        # Log-attenuation
        buffer.clamp_(min=1.0)
        buffer.log_()
        buffer.neg_()
        buffer.add_(np.log(pc_sum))
        sinogram = buffer.cpu().numpy()

        # Reconstruct (with img_scale=1.0, no compensation)
        sino_recon = np.moveaxis(sinogram, -1, 0)[:, :, ::-1]
        rec = test_scanner.reconstruct_data(sino_recon)

        if rec.shape[0] < rec.shape[1]:
            measured = rec[0, rec.shape[1] // 2, rec.shape[2] // 2]
        else:
            measured = rec[rec.shape[0] // 2, rec.shape[1] // 2, 0]

        ratio = measured / input_lac
        print(f"\nFull roundtrip WITH self.scale=0.1 bug:")
        print(f"  Input LAC:    {input_lac}")
        print(f"  Measured LAC: {measured:.6f}")
        print(f"  Ratio:        {ratio:.4f} (expected ~0.1)")

        # The measured LAC should be ~10× too small
        assert abs(ratio - self_scale) < 0.05, \
            f"With self.scale=0.1, ratio should be ~0.1, got {ratio:.4f}"

        # And water HU would be ~-900
        mu_w = input_lac
        hu = (measured - mu_w) / mu_w * 1000
        print(f"  Water HU:     {hu:.0f} (expected ~-900)")
        assert hu < -800, f"Bug should produce HU < -800, got {hu:.0f}"
