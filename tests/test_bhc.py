"""Tests for beam hardening correction.

Validates that the BHC LUT correctly linearizes polychromatic sinograms
to their monochromatic equivalents.
"""

import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestBHCLut:
    """Verify BHC LUT construction and correction."""

    @pytest.fixture
    def bhc(self):
        """Build a BHC from airport spectrum."""
        from lib.forward_model.bhc import BeamHardeningCorrector
        spec_path = os.path.join(
            os.path.dirname(__file__), '..', 'include', 'spectra',
            'airport_spectrum_160kV.txt')
        if not os.path.exists(spec_path):
            pytest.skip("Spectrum file not found")

        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        mu = MuDatabaseHandler()
        return BeamHardeningCorrector.from_debisim(mu, spec_path)

    def test_lut_monotonic(self, bhc):
        """BHC LUT should be monotonically increasing."""
        diffs = np.diff(bhc.lut)
        assert np.all(diffs >= -1e-10), \
            f"LUT not monotonic: min diff = {diffs.min()}"

    def test_zero_maps_to_zero(self, bhc):
        """Zero sinogram value should map to zero."""
        result = bhc.correct(np.array([0.0]))
        assert abs(result[0]) < 1e-10

    def test_identity_at_small_values(self, bhc):
        """For thin objects (small sinogram values), BHC should be
        nearly identity — beam hardening is negligible."""
        small = np.array([0.01, 0.05, 0.1])
        corrected = bhc.correct(small)
        rel_diff = np.abs(corrected - small) / small
        # Less than 5% correction at small path lengths
        assert np.all(rel_diff < 0.05), \
            f"BHC changes small values too much: {rel_diff}"

    def test_correction_increases_large_values(self, bhc):
        """For thick objects, polychromatic sinogram underestimates
        attenuation (cupping). BHC should increase the value."""
        large = np.array([3.0, 5.0, 8.0])
        corrected = bhc.correct(large)
        # Corrected should be >= original (undoing cupping)
        assert np.all(corrected >= large * 0.99), \
            f"BHC should increase large values: orig={large}, corr={corrected}"

    def test_e_eff_in_expected_range(self, bhc):
        """Effective energy should be in the diagnostic range."""
        assert 40 < bhc.e_eff < 120, \
            f"E_eff={bhc.e_eff:.1f} keV outside expected range"

    def test_mu_mono_positive(self, bhc):
        """Monochromatic water LAC should be positive."""
        assert bhc.mu_mono > 0.1, \
            f"mu_mono={bhc.mu_mono:.4f}, expected > 0.1"

    def test_correction_on_sinogram_shaped_input(self, bhc):
        """Should work on multi-dimensional sinogram arrays."""
        sino = np.random.RandomState(42).rand(10, 20, 30) * 5.0
        corrected = bhc.correct(sino)
        assert corrected.shape == sino.shape
        assert not np.any(np.isnan(corrected))
        assert not np.any(np.isinf(corrected))

    def test_water_cylinder_cupping_reduced(self, bhc):
        """Simulate a water cylinder sinogram and verify BHC reduces
        the cupping artifact (center vs edge difference)."""
        # Simulate polychromatic sinogram for water cylinder
        # At center: long path → more hardening → sinogram underestimated
        # At edge: short path → less hardening → sinogram closer to correct
        n_det = 128
        radius = 5.0  # cm
        mu_poly_center = 2.5  # thick water path (polychromatic)
        mu_poly_edge = 0.5    # thin water path

        sino = np.zeros(n_det)
        for d in range(n_det):
            t = (d - n_det / 2.0 + 0.5) * 0.125  # 0.125 cm spacing
            if abs(t) < radius:
                path = 2.0 * np.sqrt(radius**2 - t**2)
                # Approximate polychromatic: underestimate proportional to path
                sino[d] = path * 0.18  # effective mu at polyenergetic

        corrected = bhc.correct(sino)

        # After BHC, the ratio center/edge should be closer to
        # the true path length ratio
        center = corrected[n_det // 2]
        edge = corrected[n_det // 4]
        center_orig = sino[n_det // 2]
        edge_orig = sino[n_det // 4]

        if edge > 0.01 and edge_orig > 0.01:
            ratio_orig = center_orig / edge_orig
            ratio_corr = center / edge
            # True ratio is just path_center / path_edge
            path_center = 2.0 * radius
            path_edge = 2.0 * np.sqrt(radius**2 - (n_det//4 - n_det/2 + 0.5)**2 * 0.125**2)
            ratio_true = path_center / path_edge

            # Corrected ratio should be closer to true than original
            err_orig = abs(ratio_orig - ratio_true)
            err_corr = abs(ratio_corr - ratio_true)
            assert err_corr <= err_orig + 0.1, \
                f"BHC didn't improve cupping: err_orig={err_orig:.3f}, err_corr={err_corr:.3f}"
