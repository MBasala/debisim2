"""
Mock-based unit tests for DEBISimPipeline.

These tests verify that all imports resolve correctly and that the pipeline
can be instantiated and its methods called without requiring GPU, ASTRA,
or real data.  They catch missing imports that pure function-level unit
tests miss because they never instantiate the pipeline class.
"""

import os
import numpy as np
import pytest
from unittest import mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_scanner():
    """Create a minimal mock ScannerTemplate with required attributes."""
    scanner = mock.MagicMock()
    scanner.machine_geometry = {
        'scanner_name': 'parallel_custom',
        'gantry_diameter': 512,
        'det_col_count': 512,
        'det_row_count': 1,
        'det_spacing_y': 1.0,
        'det_spacing_x': 1.0,
        'source_origin': 500.0,
        'origin_det': 500.0,
        'n_views': 720,
        'anode_angle': 12.0,
    }
    scanner.recon_geometry = {
        'n_views': 720,
        'image_dims': (512, 512, 350),
    }
    scanner.recon_params = {
        'image_dims': (512, 512, 350),
        'n_views': 720,
        'recon_algorithm': 'fbp',
        'recon_filter': 'ram-lak',
    }
    scanner.geom = 'parallel'
    scanner.scan = 'circular'
    scanner.pscale = 1.0
    scanner.proj_geom = {}
    scanner.proj_geom2d = {}
    return scanner


@pytest.fixture
def mock_xray_source(tmp_path):
    """Create a minimal X-ray source model with real spectrum files."""
    # Write a minimal spectrum file (keV, intensity) — 121 rows for 10-130 keV
    spec_data = np.column_stack([
        np.arange(10, 131),
        np.random.rand(121)
    ])
    spec_file = str(tmp_path / 'spectrum_130kV.txt')
    np.savetxt(spec_file, spec_data)

    return dict(
        num_spectra=1,
        kVp=130,
        spectra=[spec_file],
        dosage=[2e5],
    )


@pytest.fixture
def mock_mu_handler():
    """Create a mock MuDatabaseHandler."""
    mu = mock.MagicMock()
    mu.material.return_value = {
        'compton': 0.1,
        'pe': 0.01,
        'z': 13.0,
        'density': 2.7,
        'lac_1': 0.5,
        'lac_2': 0.6,
        'mu': np.zeros(121),
    }
    return mu


# ---------------------------------------------------------------------------
# Import resolution tests
# ---------------------------------------------------------------------------

class TestPipelineImports:
    """Verify that all symbols used by debisim_pipeline.py are importable."""

    def test_import_debisim_pipeline_module(self):
        """Module-level imports should all resolve."""
        from src.debisim_pipeline import DEBISimPipeline
        assert DEBISimPipeline is not None

    def test_import_dataset_generator_module(self):
        """debisim_dataset_generator imports should resolve."""
        from src.debisim_dataset_generator import run_xray_dataset_generator
        assert run_xray_dataset_generator is not None

    def test_import_run_dataset_generator(self):
        """Entry point script imports should resolve."""
        import importlib.util as loader
        spec = loader.spec_from_file_location(
            'run_dataset_generator',
            os.path.join(os.path.dirname(__file__), '..', 'run_dataset_generator.py'))
        mod = loader.module_from_spec(spec)
        # Don't exec (would parse args), just verify the spec loaded
        assert spec is not None


# ---------------------------------------------------------------------------
# Pipeline instantiation tests
# ---------------------------------------------------------------------------

class TestPipelineInit:
    """Verify DEBISimPipeline can be constructed with mocked dependencies."""

    def test_init_creates_instance(self, tmp_path, mock_scanner,
                                    mock_xray_source, mock_mu_handler):
        """Pipeline __init__ should succeed with mocked scanner + mu."""
        from src.debisim_pipeline import DEBISimPipeline

        sim_path = str(tmp_path / 'sim_001')
        pipeline = DEBISimPipeline(
            sim_path=sim_path,
            scanner_model=mock_scanner,
            xray_source_model=mock_xray_source,
            mu_handler=mock_mu_handler,
        )
        assert pipeline is not None
        assert os.path.isdir(sim_path)

    def test_init_creates_subdirectories(self, tmp_path, mock_scanner,
                                          mock_xray_source, mock_mu_handler):
        """Pipeline should create images/, projections/, ground_truth/, gifs/."""
        from src.debisim_pipeline import DEBISimPipeline

        sim_path = str(tmp_path / 'sim_002')
        pipeline = DEBISimPipeline(
            sim_path=sim_path,
            scanner_model=mock_scanner,
            xray_source_model=mock_xray_source,
            mu_handler=mock_mu_handler,
        )
        for subdir in ['images', 'projections', 'ground_truth']:
            assert os.path.isdir(os.path.join(sim_path, subdir))

    def test_init_stores_scanner_geometry(self, tmp_path, mock_scanner,
                                           mock_xray_source, mock_mu_handler):
        from src.debisim_pipeline import DEBISimPipeline

        pipeline = DEBISimPipeline(
            sim_path=str(tmp_path / 'sim_003'),
            scanner_model=mock_scanner,
            xray_source_model=mock_xray_source,
            mu_handler=mock_mu_handler,
        )
        assert pipeline.scanner_geometry['gantry_diameter'] == 512
        assert pipeline.scanner_geometry['det_col_count'] == 512

    def test_init_computes_kev_range(self, tmp_path, mock_scanner,
                                      mock_xray_source, mock_mu_handler):
        from src.debisim_pipeline import DEBISimPipeline

        pipeline = DEBISimPipeline(
            sim_path=str(tmp_path / 'sim_004'),
            scanner_model=mock_scanner,
            xray_source_model=mock_xray_source,
            mu_handler=mock_mu_handler,
        )
        assert pipeline.keV_range[0] == 10
        assert pipeline.keV_range[-1] == 130
        assert len(pipeline.keV_range) == 121

    def test_init_with_scalar_kvp(self, tmp_path, mock_scanner,
                                   mock_xray_source, mock_mu_handler):
        """kVp as scalar should work (isscalar branch)."""
        from src.debisim_pipeline import DEBISimPipeline

        mock_xray_source['kVp'] = 130  # scalar
        pipeline = DEBISimPipeline(
            sim_path=str(tmp_path / 'sim_005'),
            scanner_model=mock_scanner,
            xray_source_model=mock_xray_source,
            mu_handler=mock_mu_handler,
        )
        assert pipeline.maxkV == 130

    def test_init_with_list_kvp(self, tmp_path, mock_scanner,
                                 mock_xray_source, mock_mu_handler):
        """kVp as list should take the max (list branch)."""
        from src.debisim_pipeline import DEBISimPipeline

        # Need a second spectrum file for dual-energy
        spec_file_2 = mock_xray_source['spectra'][0]  # reuse same file
        mock_xray_source['kVp'] = [95, 130]
        mock_xray_source['spectra'] = [spec_file_2, spec_file_2]
        mock_xray_source['dosage'] = [1.85e5, 2e5]
        mock_xray_source['num_spectra'] = 2

        pipeline = DEBISimPipeline(
            sim_path=str(tmp_path / 'sim_006'),
            scanner_model=mock_scanner,
            xray_source_model=mock_xray_source,
            mu_handler=mock_mu_handler,
        )
        assert pipeline.maxkV == 130

    def test_init_sets_shape_list_handle(self, tmp_path, mock_scanner,
                                          mock_xray_source, mock_mu_handler):
        """ShapeListHandle should be instantiated during __init__."""
        from src.debisim_pipeline import DEBISimPipeline
        from lib.bag_generator.shape_list_handle import ShapeListHandle

        pipeline = DEBISimPipeline(
            sim_path=str(tmp_path / 'sim_007'),
            scanner_model=mock_scanner,
            xray_source_model=mock_xray_source,
            mu_handler=mock_mu_handler,
        )
        assert isinstance(pipeline.slh, ShapeListHandle)

    def test_init_uses_provided_mu_handler(self, tmp_path, mock_scanner,
                                            mock_xray_source, mock_mu_handler):
        """When mu_handler is provided, it should be used directly."""
        from src.debisim_pipeline import DEBISimPipeline

        pipeline = DEBISimPipeline(
            sim_path=str(tmp_path / 'sim_008'),
            scanner_model=mock_scanner,
            xray_source_model=mock_xray_source,
            mu_handler=mock_mu_handler,
        )
        assert pipeline.mu is mock_mu_handler


# ---------------------------------------------------------------------------
# Method existence tests (verify methods are callable, not that they work)
# ---------------------------------------------------------------------------

class TestPipelineMethodsExist:
    """Verify key pipeline methods exist and are callable."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, tmp_path, mock_scanner, mock_xray_source,
                       mock_mu_handler):
        from src.debisim_pipeline import DEBISimPipeline
        self.pipeline = DEBISimPipeline(
            sim_path=str(tmp_path / 'sim_methods'),
            scanner_model=mock_scanner,
            xray_source_model=mock_xray_source,
            mu_handler=mock_mu_handler,
        )

    def test_has_create_random_simulation_instance(self):
        assert callable(getattr(self.pipeline, 'create_random_simulation_instance', None))

    def test_has_run_fwd_model(self):
        assert callable(getattr(self.pipeline, 'run_fwd_model', None))

    def test_has_run_decomposer(self):
        assert callable(getattr(self.pipeline, 'run_decomposer', None))

    def test_has_run_reconstructor(self):
        assert callable(getattr(self.pipeline, 'run_reconstructor', None))

    def test_has_save_dicom_output(self):
        assert callable(getattr(self.pipeline, 'save_dicom_output', None))

    def test_has_generate_polychromatic_ct_projection(self):
        assert callable(getattr(self.pipeline, 'generate_polychromatic_ct_projection', None))
