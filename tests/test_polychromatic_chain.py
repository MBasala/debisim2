"""
Unit tests for polychromatic energy loop consistency.

Traces every transformation in the energy accumulation chain and
verifies that mu_w (used for HU conversion) is computed with the
same weighting scheme as the energy loop sinogram accumulation.

KEY FINDING: The energy loop (line 903 in debisim_pipeline.py) weights
each keV bin by:
    w_e = spectrum[e] * e    (energy-weighted detector response)

But calculate_lac_hu_values weights by:
    w_e = spectrum[e]         (photon-count weighted)

This mismatch means the "water LAC" used for HU normalization doesn't
match the effective LAC that the energy loop produces for a water volume.
Result: water reconstructs to -640 HU instead of 0 HU.
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
def water_mu():
    """Load water mass attenuation coefficients."""
    from lib.forward_model.mu_database_handler import MuDatabaseHandler
    mu = MuDatabaseHandler()
    water = mu.material('water')
    return water['mu'], float(water['density'])


@pytest.fixture(scope="module")
def spectrum_160kv():
    """Load 160kV airport spectrum."""
    path = os.path.join('include', 'spectra', 'airport_spectrum_160kV.txt')
    if not os.path.exists(path):
        # Fall back to any available spectrum
        import glob
        specs = glob.glob(os.path.join('include', 'spectra', '*.txt'))
        if not specs:
            pytest.skip("No spectrum files available")
        path = specs[0]
    s = np.loadtxt(path)
    return s[:, 1]  # spectrum values only


# ===========================================================================
# 1. Weighting Scheme Comparison
# ===========================================================================

class TestWeightingSchemes:
    """Compare the two weighting schemes used in the pipeline."""

    def test_energy_loop_uses_energy_weighting(self):
        """The energy loop scale factor includes '* e' (keV value).

        Line 903: scale = curr_pc * curr_spectrum[e-10] * system_gain * e

        The curr_pc and system_gain are constants that cancel in the
        log normalization. But 'e' varies per keV and shifts the
        effective spectrum toward higher energies.
        """
        # Simulated weights for 3 energy bins
        spectrum = np.array([0.3, 0.5, 0.2])
        energies = np.array([50, 60, 70])  # keV
        pc = 1e5
        gain = 2.5e-3

        # Energy loop weights (what actually accumulates)
        loop_weights = spectrum * energies * pc * gain
        # After log normalization, pc and gain cancel:
        loop_weights_normalized = spectrum * energies
        loop_weights_normalized /= loop_weights_normalized.sum()

        # mu_w weights (what calculate_lac_hu_values uses)
        muw_weights = spectrum.copy()
        muw_weights /= muw_weights.sum()

        # They should be DIFFERENT
        assert not np.allclose(loop_weights_normalized, muw_weights), \
            "Energy loop and mu_w should use different weights"

        # Energy loop should weight higher energies more
        assert loop_weights_normalized[2] > muw_weights[2], \
            "Energy loop should give more weight to high-energy bin"

    def test_muw_energy_weighted(self, water_mu, spectrum_160kv):
        """calculate_lac_hu_values uses energy-weighted formula:
            LAC_w = sum(spectrum[i] * e * mu[i] * rho) / sum(spectrum[i] * e)

        This matches the energy loop's weighting scheme (line 903).
        """
        mu_arr, density = water_mu
        spec = spectrum_160kv
        n = min(len(mu_arr), len(spec))

        energies = np.arange(10, 10 + n, dtype=np.float64)
        weights = spec[:n].astype(np.float64) * energies
        lac_energy = np.sum(weights * mu_arr[:n].astype(np.float64) * density) / np.sum(weights)

        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        mu_db = MuDatabaseHandler()
        mu_db.calculate_lac_hu_values('water', [spec[:n]])
        lac_func = mu_db.material('water')['lac_1']

        np.testing.assert_allclose(lac_func, lac_energy, rtol=1e-4,
                                   err_msg="calculate_lac_hu_values should match "
                                           "energy-weighted LAC")

    def test_energy_weighted_lac_is_lower(self, water_mu, spectrum_160kv):
        """Energy-weighted LAC should be lower than photon-weighted LAC
        because higher energies (lower mu) get more weight."""
        mu_arr, density = water_mu
        spec = spectrum_160kv
        n = min(len(mu_arr), len(spec))

        # Photon-count weighted (current mu_w)
        total_photon = np.sum(spec[:n] * np.exp(-mu_arr[:n] * density))
        lac_photon = -np.log(total_photon)

        # Energy-weighted (what energy loop actually uses)
        energies = np.arange(10, 10 + n)
        w_energy = spec[:n] * energies
        total_energy = np.sum(w_energy * np.exp(-mu_arr[:n] * density))
        norm_energy = np.sum(w_energy)
        lac_energy = -np.log(total_energy / norm_energy)

        print(f"\nLAC comparison:")
        print(f"  Photon-weighted:  {lac_photon:.6f}")
        print(f"  Energy-weighted:  {lac_energy:.6f}")
        print(f"  Ratio:            {lac_photon / lac_energy:.4f}")
        print(f"  HU error:         {(lac_energy - lac_photon) / lac_photon * 1000:.0f}")

        assert lac_energy < lac_photon, \
            "Energy-weighted LAC should be lower than photon-weighted"

    def test_mismatch_causes_hu_error(self, water_mu, spectrum_160kv):
        """Using photon-weighted mu_w to normalize energy-weighted
        reconstruction gives non-zero HU for water."""
        mu_arr, density = water_mu
        spec = spectrum_160kv
        n = min(len(mu_arr), len(spec))

        # What mu_w says water LAC is
        total_photon = np.sum(spec[:n] * np.exp(-mu_arr[:n] * density))
        mu_w = -np.log(total_photon)

        # What the energy loop actually produces for a thin water sample
        energies = np.arange(10, 10 + n)
        w_energy = spec[:n] * energies
        total_energy = np.sum(w_energy * np.exp(-mu_arr[:n] * density))
        norm_energy = np.sum(w_energy)
        effective_lac = -np.log(total_energy / norm_energy)

        # HU conversion using mismatched mu_w
        hu = (effective_lac - mu_w) / mu_w * 1000

        print(f"\nHU error from weighting mismatch:")
        print(f"  mu_w (photon):    {mu_w:.6f}")
        print(f"  effective LAC:    {effective_lac:.6f}")
        print(f"  Water HU:         {hu:.0f} (expected 0)")

        # This proves the mismatch produces wrong HU
        assert abs(hu) > 50, \
            f"Expected significant HU error from mismatch, got {hu:.0f}"


# ===========================================================================
# 2. Monochromatic Consistency Check
# ===========================================================================

class TestMonochromaticConsistency:
    """Verify that monochromatic case gives correct results
    (the * e factor should cancel when spectrum has one bin)."""

    def test_single_energy_cancels_e_factor(self):
        """With a delta spectrum at one energy, the 'e' factor cancels
        in the log normalization and HU should be correct."""
        mu_at_60 = 0.2059  # water at 60 keV
        density = 1.0
        e = 60
        pc = 1e5
        gain = 2.5e-3

        # Energy loop for thin sample (proj = mu * density)
        proj = mu_at_60 * density  # = 0.2059

        scale = pc * 1.0 * gain * e  # spectrum = 1.0 for delta
        buffer = np.exp(-proj) * scale
        pc_sum = scale

        sino = -np.log(buffer) + np.log(pc_sum)

        # mu_w for monochromatic: just mu * density
        mu_w = mu_at_60 * density

        # These should match (monochromatic = no beam hardening)
        np.testing.assert_allclose(sino, proj, rtol=1e-10,
                                   err_msg="Monochromatic sinogram should equal projection")

        hu = (sino - mu_w) / mu_w * 1000
        assert abs(hu) < 1e-6, \
            f"Monochromatic water HU should be 0, got {hu:.2f}"


# ===========================================================================
# 3. Beam Hardening Quantification
# ===========================================================================

class TestBeamHardening:
    """Quantify beam hardening as a function of path length.

    Beam hardening causes the effective LAC to decrease with increasing
    path length because low-energy photons are preferentially absorbed.
    """

    def test_effective_lac_decreases_with_thickness(self, water_mu, spectrum_160kv):
        """Thicker water samples should have lower effective LAC."""
        mu_arr, density = water_mu
        spec = spectrum_160kv
        n = min(len(mu_arr), len(spec))
        energies = np.arange(10, 10 + n)

        # Use energy-weighted spectrum (matching energy loop)
        weights = spec[:n] * energies

        effective_lacs = []
        thicknesses = [1, 5, 10, 20, 50]

        print("\nBeam hardening vs thickness:")
        print(f"  {'Thickness':>10} {'Eff LAC':>10} {'Ratio to thin':>14}")

        for t in thicknesses:
            total = np.sum(weights * np.exp(-mu_arr[:n] * density * t))
            norm = np.sum(weights)
            eff_lac = -np.log(total / norm) / t
            effective_lacs.append(eff_lac)
            ratio = eff_lac / effective_lacs[0] if effective_lacs[0] > 0 else 0
            print(f"  {t:10d} {eff_lac:10.6f} {ratio:14.4f}")

        # Effective LAC should monotonically decrease
        for i in range(len(effective_lacs) - 1):
            assert effective_lacs[i] >= effective_lacs[i + 1], \
                f"Effective LAC should decrease with thickness"

    def test_thin_sample_matches_muw(self, water_mu, spectrum_160kv):
        """For a very thin sample (t -> 0), effective LAC should approach
        the energy-weighted mu_w (not photon-weighted)."""
        mu_arr, density = water_mu
        spec = spectrum_160kv
        n = min(len(mu_arr), len(spec))
        energies = np.arange(10, 10 + n)
        weights = spec[:n] * energies

        # Very thin sample
        t = 0.001
        total = np.sum(weights * np.exp(-mu_arr[:n] * density * t))
        norm = np.sum(weights)
        eff_lac_thin = -np.log(total / norm) / t

        # Energy-weighted mu_w (thin-sample limit)
        mu_w_energy = np.sum(weights * mu_arr[:n] * density) / np.sum(weights)

        print(f"\nThin-sample limit:")
        print(f"  Effective LAC (t=0.001): {eff_lac_thin:.6f}")
        print(f"  Energy-weighted mu_w:    {mu_w_energy:.6f}")

        np.testing.assert_allclose(eff_lac_thin, mu_w_energy, rtol=0.01,
                                   err_msg="Thin sample LAC should match "
                                           "energy-weighted mu_w")


# ===========================================================================
# 4. End-to-End Simulation (No ASTRA)
# ===========================================================================

class TestEndToEndNoAstra:
    """Simulate the full energy loop analytically (no GPU needed)
    to verify the accumulation math independently of ASTRA."""

    def test_water_hu_with_matched_weighting(self, water_mu, spectrum_160kv):
        """If mu_w uses the same energy-weighting as the loop,
        thin-sample water should give 0 HU."""
        mu_arr, density = water_mu
        spec = spectrum_160kv
        n = min(len(mu_arr), len(spec))
        energies = np.arange(10, 10 + n)

        # Simulate thin sample (1 voxel path)
        t = 1
        pc = 1e5
        gain = 2.5e-3

        buffer = 0.0
        pc_sum = 0.0
        for i in range(n):
            e = energies[i]
            lac = mu_arr[i] * density  # self.scale = 1.0
            proj = lac * t
            scale = pc * spec[i] * gain * e
            buffer += np.exp(-proj) * scale
            pc_sum += scale

        sino = -np.log(max(buffer, 1.0)) + np.log(pc_sum)
        effective_lac = sino / t

        # mu_w with MATCHING energy weighting
        weights = spec[:n] * energies
        mu_w_matched = np.sum(weights * mu_arr[:n] * density) / np.sum(weights)

        hu_matched = (effective_lac - mu_w_matched) / mu_w_matched * 1000

        # mu_w with MISMATCHED photon weighting (current bug)
        total_photon = np.sum(spec[:n] * np.exp(-mu_arr[:n] * density))
        mu_w_photon = -np.log(total_photon)

        hu_mismatched = (effective_lac - mu_w_photon) / mu_w_photon * 1000

        print(f"\nThin-sample water HU test:")
        print(f"  Effective LAC:     {effective_lac:.6f}")
        print(f"  mu_w (matched):    {mu_w_matched:.6f}")
        print(f"  mu_w (mismatched): {mu_w_photon:.6f}")
        print(f"  HU (matched):      {hu_matched:.1f}")
        print(f"  HU (mismatched):   {hu_mismatched:.1f}")

        assert abs(hu_matched) < 20, \
            f"Matched weighting should give HU near 0, got {hu_matched:.1f}"
        assert abs(hu_mismatched) > 50, \
            f"Mismatched weighting should give significant HU error"

    def test_air_hu_always_correct(self, water_mu, spectrum_160kv):
        """Air (no attenuation) should always give sinogram = 0,
        regardless of weighting scheme."""
        spec = spectrum_160kv
        n = len(spec)
        energies = np.arange(10, 10 + n)
        pc = 1e5
        gain = 2.5e-3

        buffer = 0.0
        pc_sum = 0.0
        for i in range(n):
            e = energies[i]
            scale = pc * spec[i] * gain * e
            buffer += 1.0 * scale  # exp(0) = 1
            pc_sum += scale

        sino = -np.log(max(buffer, 1.0)) + np.log(pc_sum)
        assert abs(sino) < 1e-10, f"Air sinogram should be 0, got {sino}"

    def test_dense_material_higher_sino_than_water(self, water_mu, spectrum_160kv):
        """A denser material should produce higher sinogram values."""
        mu_water, density_water = water_mu
        spec = spectrum_160kv
        n = min(len(mu_water), len(spec))
        energies = np.arange(10, 10 + n)
        pc = 1e5
        gain = 2.5e-3
        t = 10

        def simulate_sino(mu_arr, density):
            buf, pcs = 0.0, 0.0
            for i in range(n):
                e = energies[i]
                lac = mu_arr[i] * density
                scale = pc * spec[i] * gain * e
                buf += np.exp(-lac * t) * scale
                pcs += scale
            return -np.log(max(buf, 1.0)) + np.log(pcs)

        sino_water = simulate_sino(mu_water, density_water)

        # Simulate bone-like material (2x density, same mu curve as approx)
        sino_dense = simulate_sino(mu_water, density_water * 2.0)

        assert sino_dense > sino_water, \
            f"Dense material sino ({sino_dense:.4f}) should exceed " \
            f"water sino ({sino_water:.4f})"


# ===========================================================================
# 5. Proposed Fix Verification
# ===========================================================================

class TestProposedFix:
    """Verify that using energy-weighted mu_w fixes the HU calibration."""

    def test_energy_weighted_muw_formula(self, water_mu, spectrum_160kv):
        """The correct mu_w should be:
            mu_w = sum(spectrum[i] * e * mu[i] * density) / sum(spectrum[i] * e)

        This is the thin-sample limit of the energy loop's effective LAC.
        """
        mu_arr, density = water_mu
        spec = spectrum_160kv
        n = min(len(mu_arr), len(spec))
        energies = np.arange(10, 10 + n)

        weights = spec[:n] * energies
        mu_w_correct = np.sum(weights * mu_arr[:n] * density) / np.sum(weights)

        # Current (wrong)
        total = np.sum(spec[:n] * np.exp(-mu_arr[:n] * density))
        mu_w_current = -np.log(total)

        print(f"\nProposed mu_w fix:")
        print(f"  Current mu_w (photon-weighted):  {mu_w_current:.6f}")
        print(f"  Correct mu_w (energy-weighted):  {mu_w_correct:.6f}")
        print(f"  Difference:                      {(mu_w_current - mu_w_correct):.6f}")
        print(f"  Error in HU:                     "
              f"{(mu_w_correct - mu_w_current) / mu_w_current * 1000:.0f}")

        assert mu_w_correct < mu_w_current, \
            "Energy-weighted mu_w should be lower"
        assert mu_w_correct > 0.15, \
            f"Energy-weighted mu_w should be physically reasonable, got {mu_w_correct}"

    def test_fix_reduces_water_hu_error(self, water_mu, spectrum_160kv):
        """Using energy-weighted mu_w should reduce water HU error
        compared to photon-weighted mu_w."""
        mu_arr, density = water_mu
        spec = spectrum_160kv
        n = min(len(mu_arr), len(spec))
        energies = np.arange(10, 10 + n)
        pc = 1e5
        gain = 2.5e-3

        # Simulate thin water sample through energy loop
        t = 1
        buffer, pc_sum = 0.0, 0.0
        for i in range(n):
            e = energies[i]
            lac = mu_arr[i] * density
            scale = pc * spec[i] * gain * e
            buffer += np.exp(-lac * t) * scale
            pc_sum += scale
        sino = -np.log(max(buffer, 1.0)) + np.log(pc_sum)
        eff_lac = sino / t

        # Current mu_w (photon-weighted)
        total = np.sum(spec[:n] * np.exp(-mu_arr[:n] * density))
        mu_w_photon = -np.log(total)
        hu_current = (eff_lac - mu_w_photon) / mu_w_photon * 1000

        # Fixed mu_w (energy-weighted)
        weights = spec[:n] * energies
        mu_w_energy = np.sum(weights * mu_arr[:n] * density) / np.sum(weights)
        hu_fixed = (eff_lac - mu_w_energy) / mu_w_energy * 1000

        print(f"\nFix comparison (thin sample):")
        print(f"  Current HU: {hu_current:.1f}")
        print(f"  Fixed HU:   {hu_fixed:.1f}")

        assert abs(hu_fixed) < abs(hu_current), \
            f"Fixed HU ({hu_fixed:.0f}) should be closer to 0 than " \
            f"current ({hu_current:.0f})"
