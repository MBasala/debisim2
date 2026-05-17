"""
Integration test: run a minimal single-bag simulation end-to-end.

This exercises the full pipeline path:
  bag generation → forward projection → sinogram save → BHC →
  FBP reconstruction → DICOM export → RT-Struct

Uses the smallest possible phantom (1×1 water cylinder, 16×16×4 volume,
32 views) to keep runtime under ~60 seconds on a machine with a CUDA GPU.

Marked @pytest.mark.slow — deselect with: pytest -m "not slow"
"""

import os
import sys
import glob
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Guard: skip entire module if no CUDA GPU available
torch = pytest.importorskip('torch')
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required for integration tests", allow_module_level=True)

from lib.__init__ import SPECTRA_DIR
from lib.misc.util import read_fits_data, flush_async_io


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim_output(tmp_path_factory):
    """Run a minimal single-bag pipeline and return the output directory.

    Module-scoped so the simulation runs once and all tests in this module
    share the result.
    """
    from lib.forward_model.scanner_template import create_parallel_scanner
    from src.debisim_dataset_generator import run_xray_dataset_generator

    out_dir = str(tmp_path_factory.mktemp("mini_sim"))

    scanner = create_parallel_scanner(
        gantry_diameter_mm=128,
        pixel_size_mm=2.0,
        n_slices=4,
        n_views=32,
    )

    spec_160 = os.path.join(SPECTRA_DIR, 'airport_spectrum_160kV.txt')
    spec_80 = os.path.join(SPECTRA_DIR, 'airport_spectrum_80kV.txt')
    if not os.path.exists(spec_160) or not os.path.exists(spec_80):
        pytest.skip("Airport spectrum files not found")

    xray_source = dict(
        num_spectra=2,
        kVp=160,
        spectra=[spec_160, spec_80],
        dosage=[1e4, 8e3],       # very low dose — speed over quality
    )

    bag_args = dict(
        phantom_mode=True,
        phantom_materials=['water'],
        phantom_grid=(1, 1),
        phantom_block_size=10,
        phantom_gap=0,
        material_list=['water'],
        liquid_list=[],
        material_pdf=[1.0],
        liquid_pdf=[],
        prevent_overlap=False,
    )

    cdm_args = dict(
        cdm_solver='gpu',
        cdm_type='cpd',
        projector='gpu',
        init_val=(0.1, 0.1),
    )

    run_xray_dataset_generator(
        num_bags=range(1, 2),
        sim_dir=out_dir,
        scanner=scanner,
        xray_src_mdl=xray_source,
        bag_creator_args=bag_args,
        decomposer='cdm',
        save_sino=True,
        decomposer_args=cdm_args,
        images_to_save=['gt', 'lac_1', 'lac_2'],
        slicewise=False,
        compress_data=False,
        num_workers=1,
        dicom_output=True,
        compress_dicom=False,
    )

    # Flush any remaining async I/O before tests read outputs
    flush_async_io()

    return out_dir


@pytest.fixture(scope="module")
def sim_dir(sim_output):
    """Return the simulation_001 subdirectory."""
    d = os.path.join(sim_output, 'simulation_001')
    assert os.path.isdir(d), f"Expected {d} to exist after pipeline run"
    return d


# ===========================================================================
# Pipeline output structure
# ===========================================================================

@pytest.mark.slow
class TestOutputStructure:
    """Verify the pipeline creates all expected output directories and files."""

    def test_simulation_directory_exists(self, sim_dir):
        assert os.path.isdir(sim_dir)

    def test_sino_directory_exists(self, sim_dir):
        sino_dir = os.path.join(sim_dir, 'projections')
        assert os.path.isdir(sino_dir)

    def test_images_directory_exists(self, sim_dir):
        img_dir = os.path.join(sim_dir, 'images')
        assert os.path.isdir(img_dir)

    def test_sinograms_saved(self, sim_dir):
        """Dual-energy should produce sino_1.fits.gz and sino_2.fits.gz."""
        sino_dir = os.path.join(sim_dir, 'projections')
        sino_files = glob.glob(os.path.join(sino_dir, 'sino_*.fits*'))
        assert len(sino_files) >= 2, f"Expected ≥2 sinogram files, found: {sino_files}"

    def test_recon_images_saved(self, sim_dir):
        """Reconstructed images should exist in the images/ directory."""
        img_dir = os.path.join(sim_dir, 'images')
        recon_files = glob.glob(os.path.join(img_dir, 'recon_image_*.fits*'))
        assert len(recon_files) >= 1, f"No recon images found in {img_dir}"

    def test_gt_label_image_saved(self, sim_dir):
        """Ground truth label image should be saved."""
        img_dir = os.path.join(sim_dir, 'images')
        gt_files = glob.glob(os.path.join(img_dir, '*gt*label*.fits*'))
        assert len(gt_files) >= 1, f"No GT label image found in {img_dir}"


# ===========================================================================
# Sinogram integrity
# ===========================================================================

@pytest.mark.slow
class TestSinogramIntegrity:
    """Verify sinograms are valid numpy arrays with expected properties."""

    def test_sinogram_is_loadable(self, sim_dir):
        sino_dir = os.path.join(sim_dir, 'projections')
        sino_files = glob.glob(os.path.join(sino_dir, 'sino_1*'))
        assert len(sino_files) > 0
        data = read_fits_data(sino_files[0], 0)
        assert data is not None
        assert data.ndim == 3

    def test_sinogram_not_all_zeros(self, sim_dir):
        sino_dir = os.path.join(sim_dir, 'projections')
        sino_files = glob.glob(os.path.join(sino_dir, 'sino_1*'))
        data = read_fits_data(sino_files[0], 0)
        assert np.any(data != 0), "Sinogram is all zeros"

    def test_sinogram_has_no_nan(self, sim_dir):
        sino_dir = os.path.join(sim_dir, 'projections')
        sino_files = glob.glob(os.path.join(sino_dir, 'sino_1*'))
        data = read_fits_data(sino_files[0], 0)
        assert not np.any(np.isnan(data)), "Sinogram contains NaN"

    def test_sinogram_has_no_inf(self, sim_dir):
        sino_dir = os.path.join(sim_dir, 'projections')
        sino_files = glob.glob(os.path.join(sino_dir, 'sino_1*'))
        data = read_fits_data(sino_files[0], 0)
        assert not np.any(np.isinf(data)), "Sinogram contains Inf"


# ===========================================================================
# Reconstruction integrity
# ===========================================================================

@pytest.mark.slow
class TestReconIntegrity:
    """Verify reconstructed images have expected properties."""

    def test_recon_image_loadable(self, sim_dir):
        img_dir = os.path.join(sim_dir, 'images')
        recon_files = glob.glob(os.path.join(img_dir, 'recon_image_*.fits*'))
        assert len(recon_files) > 0
        data = read_fits_data(recon_files[0], 0)
        assert data is not None
        assert data.ndim == 3

    def test_recon_image_has_no_nan(self, sim_dir):
        img_dir = os.path.join(sim_dir, 'images')
        recon_files = glob.glob(os.path.join(img_dir, 'recon_image_*.fits*'))
        data = read_fits_data(recon_files[0], 0)
        assert not np.any(np.isnan(data)), "Recon image contains NaN"

    def test_recon_has_contrast(self, sim_dir):
        """Water cylinder should produce non-uniform HU values
        (background air vs water region)."""
        img_dir = os.path.join(sim_dir, 'images')
        recon_files = glob.glob(os.path.join(img_dir, 'recon_image_*.fits*'))
        data = read_fits_data(recon_files[0], 0)
        # At least some variation between min and max
        dynamic_range = float(np.max(data) - np.min(data))
        assert dynamic_range > 10.0, \
            f"Recon image has no contrast (range={dynamic_range:.1f})"


# ===========================================================================
# DICOM output
# ===========================================================================

@pytest.mark.slow
class TestDicomOutput:
    """Verify DICOM files are created and structurally valid."""

    def test_dicom_directory_exists(self, sim_dir):
        img_dir = os.path.join(sim_dir, 'images')
        dicom_dir = os.path.join(img_dir, 'dicom')
        assert os.path.isdir(dicom_dir), \
            f"DICOM output directory not found at {dicom_dir}"

    def test_dicom_has_dcm_files(self, sim_dir):
        img_dir = os.path.join(sim_dir, 'images')
        dicom_dir = os.path.join(img_dir, 'dicom')
        if not os.path.isdir(dicom_dir):
            pytest.skip("No DICOM directory")
        dcm_files = []
        for root, dirs, files in os.walk(dicom_dir):
            dcm_files.extend(f for f in files if f.endswith('.dcm'))
        assert len(dcm_files) > 0, "No .dcm files in DICOM output"

    def test_dicom_files_are_readable(self, sim_dir):
        """Every .dcm file should be parseable by pydicom."""
        import pydicom
        img_dir = os.path.join(sim_dir, 'images')
        dicom_dir = os.path.join(img_dir, 'dicom')
        if not os.path.isdir(dicom_dir):
            pytest.skip("No DICOM directory")
        dcm_files = []
        for root, dirs, files in os.walk(dicom_dir):
            for f in files:
                if f.endswith('.dcm'):
                    dcm_files.append(os.path.join(root, f))
        assert len(dcm_files) > 0
        for fpath in dcm_files:
            ds = pydicom.dcmread(fpath)
            assert ds is not None

    def test_rtstruct_exists(self, sim_dir):
        """An RT-Struct file should be generated alongside the CT series."""
        img_dir = os.path.join(sim_dir, 'images')
        dicom_dir = os.path.join(img_dir, 'dicom')
        if not os.path.isdir(dicom_dir):
            pytest.skip("No DICOM directory")
        rtstruct_files = []
        for root, dirs, files in os.walk(dicom_dir):
            rtstruct_files.extend(
                f for f in files if 'rtstruct' in f.lower())
        assert len(rtstruct_files) > 0, "No RT-Struct file in DICOM output"


# ===========================================================================
# No disk round-trips
# ===========================================================================

@pytest.mark.slow
class TestNoRoundTrips:
    """Verify the pipeline didn't leave evidence of unnecessary I/O."""

    def test_no_stale_temp_files(self, sim_dir):
        """No .bin or .prm temp files should remain (FreeCT artifacts)."""
        for root, dirs, files in os.walk(sim_dir):
            for f in files:
                assert not f.endswith('.bin'), f"Stale temp file: {f}"
                assert not f.endswith('.prm'), f"Stale temp file: {f}"

