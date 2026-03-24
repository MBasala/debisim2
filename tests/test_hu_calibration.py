"""
Tests for HU calibration correctness.

Verifies that the HU conversion produces physically correct values
for known materials, regardless of scanner pixel_size_mm.
"""

import numpy as np
import pytest
from unittest import mock


class TestHUCalibration:
    """Verify that water LAC values used for HU conversion match
    the actual spectra, not hardcoded stale values."""

    def test_custom_scanner_mu_w_is_none(self):
        """create_parallel_scanner should set mu_w=None so the pipeline
        computes water LAC from actual spectra."""
        from lib.forward_model.scanner_template import create_parallel_scanner
        scanner = create_parallel_scanner(
            gantry_diameter_mm=1024, pixel_size_mm=2.0, n_slices=350)
        assert scanner.recon_params['mu_w'] is None

    def test_custom_scanner_mu_w_none_at_different_pixel_sizes(self):
        """mu_w should be None regardless of pixel_size_mm."""
        from lib.forward_model.scanner_template import create_parallel_scanner
        for px in [0.5, 1.0, 2.0, 4.0]:
            scanner = create_parallel_scanner(
                gantry_diameter_mm=1024, pixel_size_mm=px, n_slices=100)
            assert scanner.recon_params['mu_w'] is None, \
                f"mu_w not None for pixel_size_mm={px}"

    def test_computed_water_lac_is_positive(self):
        """Water LAC values computed from spectra should be positive."""
        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        import os
        mu = MuDatabaseHandler()
        spectra_dir = os.path.join(
            os.path.dirname(__file__), '..', 'include', 'spectra')
        spec_130 = os.path.join(spectra_dir, 'example_spectrum_130kV.txt')
        spec_95 = os.path.join(spectra_dir, 'example_spectrum_95kV.txt')

        if not os.path.exists(spec_130):
            pytest.skip("Spectrum files not available")

        spectra = [np.loadtxt(spec_130)[:120, 1],
                   np.loadtxt(spec_95)[:85, 1]]
        mu.calculate_lac_hu_values('water', spectra)
        w = mu.material('water')
        assert w['lac_1'] > 0, f"lac_1 should be positive, got {w['lac_1']}"
        assert w['lac_2'] > 0, f"lac_2 should be positive, got {w['lac_2']}"

    def test_computed_water_lac_in_expected_range(self):
        """Water LAC at diagnostic CT energies should be ~0.15-0.25 cm^-1."""
        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        import os
        mu = MuDatabaseHandler()
        spectra_dir = os.path.join(
            os.path.dirname(__file__), '..', 'include', 'spectra')
        spec_130 = os.path.join(spectra_dir, 'example_spectrum_130kV.txt')
        spec_95 = os.path.join(spectra_dir, 'example_spectrum_95kV.txt')

        if not os.path.exists(spec_130):
            pytest.skip("Spectrum files not available")

        spectra = [np.loadtxt(spec_130)[:120, 1],
                   np.loadtxt(spec_95)[:85, 1]]
        mu.calculate_lac_hu_values('water', spectra)
        w = mu.material('water')
        assert 0.10 < w['lac_1'] < 0.30, \
            f"lac_1={w['lac_1']:.4f} outside expected range [0.10, 0.30]"
        assert 0.10 < w['lac_2'] < 0.30, \
            f"lac_2={w['lac_2']:.4f} outside expected range [0.10, 0.30]"

    def test_hardcoded_mu_w_does_not_match_computed(self):
        """Proves that the old hardcoded mu_w values are WRONG for these
        spectra — this is the root cause of the HU bias we observed."""
        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        from lib.forward_model.scanner_template import default_scanner_parallel
        import os

        mu = MuDatabaseHandler()
        spectra_dir = os.path.join(
            os.path.dirname(__file__), '..', 'include', 'spectra')
        spec_130 = os.path.join(spectra_dir, 'example_spectrum_130kV.txt')
        spec_95 = os.path.join(spectra_dir, 'example_spectrum_95kV.txt')

        if not os.path.exists(spec_130):
            pytest.skip("Spectrum files not available")

        spectra = [np.loadtxt(spec_130)[:120, 1],
                   np.loadtxt(spec_95)[:85, 1]]
        mu.calculate_lac_hu_values('water', spectra)
        computed = mu.material('water')

        hardcoded = default_scanner_parallel.recon_params['mu_w']
        assert hardcoded is not None, "default scanner should have hardcoded mu_w"

        # These should NOT match — proving the hardcoded values are stale
        assert abs(computed['lac_1'] - hardcoded['lac_1']) > 0.01, \
            "Hardcoded lac_1 unexpectedly matches computed — test premise invalid"

    def test_img_scale_is_unity(self):
        """img_scale should be 1.0 for all pixel sizes.
        Unit correction is now handled via sino_correction (1/self.scale)
        applied to the sinogram before reconstruction."""
        from lib.forward_model.scanner_template import create_parallel_scanner
        for px in [0.5, 1.0, 2.0]:
            scanner = create_parallel_scanner(
                gantry_diameter_mm=512, pixel_size_mm=px, n_slices=100)
            actual_scale = scanner.recon_params['img_scale']
            assert abs(actual_scale - 1.0) < 1e-10, \
                f"img_scale={actual_scale} != 1.0 for pixel_size_mm={px}"
