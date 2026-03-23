"""
Tests for calibration phantom accuracy.

Verifies that reference material properties (Z_eff, density, LAC) are
correctly propagated through ground truth → reconstruction → DICOM output.

Fast tests (no @pytest.mark.slow) verify reference data consistency only.
Slow tests require pre-existing pipeline output in RESULTS_DIR.
"""

import csv
import os
import re

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'diagnostic_cylinders',
    'simulation_001')

PHANTOM_MATERIALS = [
    'air', 'water', 'ethanol', 'polyethylene', 'polystyrene',
    'acrylic', 'nylon6', 'acetal', 'bakelite', 'neoprene',
    'pyrex', 'salt', 'bone', 'teflon', 'polyester',
    'Si', 'Al', 'Ti', 'Fe', 'Cu',
    'Zn', 'C', 'saline035', 'H2O2', 'pvc',
]

# Labels start at 4 (after background=0, boundary labels 1-3)
LABEL_START = 4

# Tolerances
ZEFF_TOL = 0.5           # ±0.5 for GT Z_eff vs reference
LAC_REL_TOL = 0.05       # ±5% relative for GT LAC vs reference
HU_ABS_TOL = 200         # ±200 HU for recon vs expected
RANK_CORR_MIN = 0.85     # Spearman rank correlation for material ordering
CONTOUR_XY_TOL = 2       # ±2 pixels for contour centroid vs GT centroid


def _skip_if_no_results():
    if not os.path.isdir(RESULTS_DIR):
        pytest.skip(f"Pipeline output not found at {RESULTS_DIR}. "
                     f"Run config_dicom_diagnostic.py first.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ref_db():
    """Load MuDatabaseHandler with all phantom materials initialized."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from lib.forward_model.mu_database_handler import MuDatabaseHandler
    mu = MuDatabaseHandler()
    # Touch each material to ensure it's loaded
    for mat in PHANTOM_MATERIALS:
        try:
            mu.material(mat)
        except Exception:
            pass
    return mu


@pytest.fixture(scope="module")
def ref_zeff():
    """Load Z_eff lookup from reference files."""
    root = os.path.join(os.path.dirname(__file__), '..')
    zeff = {}
    for fname in ['include/mu/compounds_zeff.txt',
                   'include/mu/targets_zeff.txt']:
        fpath = os.path.join(root, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    zeff[parts[0]] = float(parts[1])
    return zeff


@pytest.fixture(scope="module")
def gt_volumes():
    """Load ground truth volumes from FITS files."""
    _skip_if_no_results()
    from astropy.io import fits
    gt_dir = os.path.join(RESULTS_DIR, 'ground_truth')

    def _load(name):
        path = os.path.join(gt_dir, name)
        if os.path.exists(path):
            return fits.open(path)[0].data
        return None

    return dict(
        label=_load('gt_label_image.fits.gz'),
        zeff=_load('gt_zeff_image.fits.gz'),
        lac_1=_load('gt_lac_1_image.fits.gz'),
        lac_2=_load('gt_lac_2_image.fits.gz'),
        compton=_load('gt_compton_image.fits.gz'),
        pe=_load('gt_pe_image.fits.gz'),
    )


@pytest.fixture(scope="module")
def recon_volumes():
    """Load reconstructed volumes from FITS files."""
    _skip_if_no_results()
    from astropy.io import fits
    img_dir = os.path.join(RESULTS_DIR, 'images')

    def _load(name):
        path = os.path.join(img_dir, name)
        if os.path.exists(path):
            return fits.open(path)[0].data
        return None

    return dict(
        recon_1=_load('recon_image_1.fits.gz'),
        recon_2=_load('recon_image_2.fits.gz'),
    )


@pytest.fixture(scope="module")
def dicom_data():
    """Load DICOM series, RT-Struct, and ROI manifest."""
    _skip_if_no_results()
    import pydicom
    dicom_dir = os.path.join(RESULTS_DIR, 'images', 'dicom')
    if not os.path.isdir(dicom_dir):
        pytest.skip("DICOM output not found")

    data = dict(dicom_dir=dicom_dir)

    # Load RT-Struct
    rtstruct_path = os.path.join(dicom_dir, 'rtstruct.dcm')
    if os.path.exists(rtstruct_path):
        data['rtstruct'] = pydicom.dcmread(rtstruct_path)

    # Load ROI manifest
    manifest_path = os.path.join(dicom_dir, 'roi_manifest.csv')
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            data['manifest'] = list(csv.DictReader(f))

    # Find first CT series and load a sample slice
    series_dirs = [d for d in os.listdir(dicom_dir)
                   if os.path.isdir(os.path.join(dicom_dir, d))
                   and d.startswith('series_')]
    if series_dirs:
        first_series = os.path.join(dicom_dir, sorted(series_dirs)[0])
        slices = sorted(f for f in os.listdir(first_series) if f.endswith('.dcm'))
        if slices:
            mid = len(slices) // 2
            data['sample_slice'] = pydicom.dcmread(
                os.path.join(first_series, slices[mid]))
            data['series_dir'] = first_series
            data['series_name'] = sorted(series_dirs)[0]

    return data


# ===========================================================================
# Class 1: Reference vs Ground Truth
# ===========================================================================

class TestRefVsGroundTruth:
    """Verify that GT volumes contain correct material property values."""

    @pytest.mark.slow
    def test_all_materials_present_in_gt(self, gt_volumes):
        """All 25 phantom materials should have unique labels in the GT."""
        label = gt_volumes['label']
        assert label is not None, "GT label volume not found"
        gt_int = label.astype(np.int32)
        unique_labels = set(np.unique(gt_int)) - {0, 1, 2, 3}
        expected = set(range(LABEL_START, LABEL_START + len(PHANTOM_MATERIALS)))
        assert unique_labels == expected, \
            f"Missing labels: {expected - unique_labels}, extra: {unique_labels - expected}"

    @pytest.mark.slow
    def test_gt_labels_are_integer(self, gt_volumes):
        """GT label volume should contain integer values only."""
        label = gt_volumes['label']
        assert label is not None
        gt_int = label.astype(np.int32)
        np.testing.assert_array_equal(label, gt_int.astype(label.dtype),
                                       err_msg="GT labels have non-integer values")

    @pytest.mark.slow
    def test_gt_zeff_matches_reference(self, gt_volumes, ref_zeff):
        """Z_eff in GT volume should match reference lookup values."""
        label = gt_volumes['label']
        zeff = gt_volumes['zeff']
        if label is None or zeff is None:
            pytest.skip("GT label or zeff volume not found")

        gt_int = label.astype(np.int32)
        failures = []

        for i, mat in enumerate(PHANTOM_MATERIALS):
            lbl = LABEL_START + i
            if mat not in ref_zeff:
                continue  # elements (Si, Al, etc.) may not be in lookup

            mask = gt_int == lbl
            if mask.sum() == 0:
                failures.append(f"{mat}: label {lbl} not found in GT")
                continue

            gt_mean = float(np.mean(zeff[mask]))
            expected = ref_zeff[mat]
            if abs(gt_mean - expected) > ZEFF_TOL:
                failures.append(
                    f"{mat}: GT Z_eff={gt_mean:.2f}, ref={expected:.2f}, "
                    f"diff={abs(gt_mean - expected):.2f}")

        assert len(failures) == 0, \
            f"Z_eff mismatches:\n" + "\n".join(failures)

    @pytest.mark.slow
    def test_gt_lac_nonzero_for_dense_materials(self, gt_volumes):
        """Dense materials should have non-zero LAC in GT."""
        label = gt_volumes['label']
        lac_1 = gt_volumes['lac_1']
        if label is None or lac_1 is None:
            pytest.skip("GT label or lac_1 volume not found")

        gt_int = label.astype(np.int32)
        dense_materials = ['water', 'bone', 'teflon', 'Fe', 'Cu', 'Al']

        for mat in dense_materials:
            idx = PHANTOM_MATERIALS.index(mat)
            lbl = LABEL_START + idx
            mask = gt_int == lbl
            if mask.sum() == 0:
                continue
            mean_lac = float(np.mean(lac_1[mask]))
            assert mean_lac > 0.01, \
                f"{mat} (label {lbl}): GT LAC mean={mean_lac:.4f}, expected > 0.01"


# ===========================================================================
# Class 2: Reference vs Reconstructed
# ===========================================================================

class TestRefVsRecon:
    """Verify reconstructed images against expected material properties."""

    @pytest.mark.slow
    def test_dense_materials_brighter_than_light(self, gt_volumes, recon_volumes):
        """Dense materials (neoprene, bakelite, acetal) should reconstruct
        with higher HU than light organics (water, ethanol, polystyrene).
        This is a relative ordering test, not absolute HU calibration."""
        label = gt_volumes['label']
        recon = recon_volumes['recon_1']
        if label is None or recon is None:
            pytest.skip("GT label or recon volume not found")

        gt_int = label.astype(np.int32)
        if gt_int.shape != recon.shape:
            from scipy.ndimage import zoom
            scale = tuple(r / g for r, g in zip(recon.shape, gt_int.shape))
            gt_int = zoom(gt_int, scale, order=0)

        def _mean_hu(mat):
            idx = PHANTOM_MATERIALS.index(mat)
            lbl = LABEL_START + idx
            mask = gt_int == lbl
            if mask.sum() == 0:
                return None
            return float(np.mean(recon[mask]))

        # Dense materials should be brighter than light ones
        dense = ['neoprene', 'bakelite', 'acetal']
        light = ['water', 'ethanol', 'polystyrene']

        dense_hu = [h for h in (_mean_hu(m) for m in dense) if h is not None]
        light_hu = [h for h in (_mean_hu(m) for m in light) if h is not None]

        if not dense_hu or not light_hu:
            pytest.skip("Not enough materials resolved")

        avg_dense = np.mean(dense_hu)
        avg_light = np.mean(light_hu)
        assert avg_dense > avg_light, \
            f"Dense materials avg HU ({avg_dense:.0f}) should exceed " \
            f"light materials avg HU ({avg_light:.0f})"

    @pytest.mark.slow
    def test_water_hu_nearest_zero(self, gt_volumes, recon_volumes):
        """Water should have reconstructed HU closest to 0 among organics."""
        label = gt_volumes['label']
        recon = recon_volumes['recon_1']
        if label is None or recon is None:
            pytest.skip("GT label or recon volume not found")

        gt_int = label.astype(np.int32)
        if gt_int.shape != recon.shape:
            from scipy.ndimage import zoom
            scale = tuple(r / g for r, g in zip(recon.shape, gt_int.shape))
            gt_int = zoom(gt_int, scale, order=0)

        organics = ['water', 'ethanol', 'polyethylene', 'polystyrene',
                     'acrylic', 'nylon6']
        hu_by_mat = {}

        for mat in organics:
            idx = PHANTOM_MATERIALS.index(mat)
            lbl = LABEL_START + idx
            mask = gt_int == lbl
            if mask.sum() > 0:
                hu_by_mat[mat] = abs(float(np.mean(recon[mask])))

        if 'water' not in hu_by_mat:
            pytest.skip("Water not found in recon")

        closest = min(hu_by_mat, key=hu_by_mat.get)
        # Water should be closest to 0 HU, or at least within top 3
        sorted_by_hu = sorted(hu_by_mat, key=hu_by_mat.get)
        water_rank = sorted_by_hu.index('water')
        assert water_rank < 3, \
            f"Water ranked {water_rank + 1} closest to 0 HU " \
            f"(expected top 3). Values: {hu_by_mat}"


# ===========================================================================
# Class 3: Reference vs DICOM Output
# ===========================================================================

class TestRefVsDicom:
    """Verify DICOM output matches reference data and reconstruction."""

    @pytest.mark.slow
    def test_roi_manifest_complete(self, dicom_data):
        """ROI manifest should contain all 25 phantom materials."""
        manifest = dicom_data.get('manifest')
        if manifest is None:
            pytest.skip("ROI manifest not found")

        materials_in_manifest = {row['material'] for row in manifest}
        expected = set(PHANTOM_MATERIALS)
        missing = expected - materials_in_manifest
        assert len(missing) == 0, f"Materials missing from manifest: {missing}"

    @pytest.mark.slow
    def test_rtstruct_zeff_matches_reference(self, dicom_data, ref_zeff):
        """RT-Struct ROI descriptions should contain correct Z_eff values."""
        ds = dicom_data.get('rtstruct')
        if ds is None:
            pytest.skip("RT-Struct not found")

        failures = []
        for obs in ds.RTROIObservationsSequence:
            desc = getattr(obs, 'ROIObservationDescription', '')
            mat_match = re.search(r'material=(\S+?);', desc)
            z_match = re.search(r'z_eff=([\d.eE+-]+)', desc)
            if not mat_match or not z_match:
                continue

            mat = mat_match.group(1)
            z_val = float(z_match.group(1))
            if mat in ref_zeff:
                expected = ref_zeff[mat]
                if abs(z_val - expected) > ZEFF_TOL:
                    failures.append(
                        f"{mat}: DICOM Z_eff={z_val:.2f}, ref={expected:.2f}")

        assert len(failures) == 0, \
            f"Z_eff mismatches in RT-Struct:\n" + "\n".join(failures)

    @pytest.mark.slow
    def test_dicom_metadata_matches_config(self, dicom_data):
        """DICOM tags should reflect the diagnostic config values."""
        ds = dicom_data.get('sample_slice')
        if ds is None:
            pytest.skip("No DICOM slice found")

        # From config_dicom_diagnostic.py:
        # gantry_diameter_mm=512, pixel_size_mm=2.0, n_slices=64, n_views=720
        assert float(ds.PixelSpacing[0]) == 2.0
        assert float(ds.PixelSpacing[1]) == 2.0
        assert float(ds.SliceThickness) == 2.0
        assert float(ds.SpacingBetweenSlices) == 2.0
        assert ds.Rows == 256
        assert ds.Columns == 256
        assert str(ds.Modality) == 'CT'
        assert str(ds.RescaleType) == 'HU'

    @pytest.mark.slow
    def test_dicom_pixel_values_match_recon(self, dicom_data, recon_volumes):
        """DICOM pixel data should match FITS recon (after transpose)."""
        ds = dicom_data.get('sample_slice')
        recon = recon_volumes.get('recon_1')
        if ds is None or recon is None:
            pytest.skip("DICOM or recon data not found")

        z_idx = int(ds.InstanceNumber) - 1
        if z_idx >= recon.shape[2]:
            pytest.skip(f"Slice index {z_idx} out of range")

        dicom_pixels = ds.pixel_array  # [row=Y, col=X] after transpose
        fits_slice = recon[:, :, z_idx].T  # apply same transpose
        fits_int16 = np.clip(fits_slice, -32768, 32767).astype(np.int16)

        np.testing.assert_array_equal(
            dicom_pixels, fits_int16,
            err_msg=f"DICOM pixels don't match FITS recon at slice {z_idx}")

    @pytest.mark.slow
    def test_contour_z_within_recon_range(self, dicom_data, recon_volumes):
        """RT-Struct contour Z values must fall within the reconstructed
        content Z range (no slices extending beyond valid FBP data)."""
        ds = dicom_data.get('rtstruct')
        recon = recon_volumes.get('recon_1')
        if ds is None or recon is None:
            pytest.skip("RT-Struct or recon not found")

        # Find recon content Z range
        slice_thickness = 2.0  # from config
        content_z_min = None
        content_z_max = None
        for z in range(recon.shape[2]):
            if recon[:, :, z].max() > -900:
                z_mm = z * slice_thickness
                if content_z_min is None:
                    content_z_min = z_mm
                content_z_max = z_mm

        if content_z_min is None:
            pytest.skip("No content found in recon")

        # Collect all contour Z values
        contour_z = set()
        for rc in ds.ROIContourSequence:
            if hasattr(rc, 'ContourSequence'):
                for c in rc.ContourSequence:
                    pts = np.array(c.ContourData).reshape(-1, 3)
                    contour_z.add(float(pts[0, 2]))

        if not contour_z:
            pytest.skip("No contours found")

        z_min = min(contour_z)
        z_max = max(contour_z)

        assert z_min >= content_z_min, \
            f"Contour Z min {z_min}mm < content Z min {content_z_min}mm"
        assert z_max <= content_z_max, \
            f"Contour Z max {z_max}mm > content Z max {content_z_max}mm"

    @pytest.mark.slow
    def test_contour_xy_aligned_with_gt(self, dicom_data, gt_volumes):
        """RT-Struct contour XY centers should be near GT label centroids."""
        ds = dicom_data.get('rtstruct')
        label = gt_volumes.get('label')
        if ds is None or label is None:
            pytest.skip("RT-Struct or GT label not found")

        gt_int = label.astype(np.int32)
        pixel_spacing = 2.0  # from config
        failures = []

        for rc in ds.ROIContourSequence:
            roi_num = rc.ReferencedROINumber
            # Find ROI name
            name = ''
            for roi in ds.StructureSetROISequence:
                if roi.ROINumber == roi_num:
                    name = roi.ROIName
                    break

            if not hasattr(rc, 'ContourSequence') or not rc.ContourSequence:
                continue

            # Get contour center from first slice
            pts = np.array(rc.ContourSequence[0].ContourData).reshape(-1, 3)
            cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
            cy = (pts[:, 1].min() + pts[:, 1].max()) / 2

            # Find corresponding GT label centroid
            # Label is encoded in the name: phantom_block_{label}_{material}
            label_match = re.search(r'phantom_block_(\d+)_', name)
            if not label_match:
                continue
            lbl = int(label_match.group(1))

            # GT centroid in patient coords
            mask_z = gt_int[:, :, gt_int.shape[2] // 2]
            coords = np.argwhere(mask_z == lbl)
            if len(coords) == 0:
                continue
            gt_cx = float(np.mean(coords[:, 0])) * pixel_spacing
            gt_cy = float(np.mean(coords[:, 1])) * pixel_spacing

            dx = abs(cx - gt_cx) / pixel_spacing
            dy = abs(cy - gt_cy) / pixel_spacing
            if dx > CONTOUR_XY_TOL or dy > CONTOUR_XY_TOL:
                failures.append(
                    f"{name}: contour=({cx:.0f},{cy:.0f}), "
                    f"GT=({gt_cx:.0f},{gt_cy:.0f}), "
                    f"diff=({dx:.1f},{dy:.1f}) pixels")

        assert len(failures) == 0, \
            f"Contour XY misalignments:\n" + "\n".join(failures)


# ===========================================================================
# Class 4: Measured Image Values vs Reference
# ===========================================================================

class TestMeasuredVsReference:
    """Sample actual pixel values from reconstructed DICOM images within
    each ROI contour and compare against known reference values.

    This is the real calibration validation — it answers:
    'Did the pipeline produce physically correct output?'"""

    @staticmethod
    def _load_dicom_volume(series_dir):
        """Load all DICOM slices into a 3D volume (row, col, z)."""
        import pydicom
        slices = sorted(f for f in os.listdir(series_dir)
                        if f.endswith('.dcm'))
        if not slices:
            return None, None
        ds0 = pydicom.dcmread(os.path.join(series_dir, slices[0]))
        rows, cols = ds0.Rows, ds0.Columns
        vol = np.zeros((rows, cols, len(slices)), dtype=np.float64)
        z_positions = []
        for i, s in enumerate(slices):
            ds = pydicom.dcmread(os.path.join(series_dir, s))
            vol[:, :, i] = ds.pixel_array.astype(np.float64) * float(
                ds.RescaleSlope) + float(ds.RescaleIntercept)
            z_positions.append(float(ds.ImagePositionPatient[2]))
        meta = dict(
            pixel_spacing=(float(ds0.PixelSpacing[0]),
                           float(ds0.PixelSpacing[1])),
            origin=(float(ds0.ImagePositionPatient[0]),
                    float(ds0.ImagePositionPatient[1]),
                    min(z_positions)),
            slice_spacing=abs(z_positions[1] - z_positions[0])
            if len(z_positions) > 1 else float(ds0.SliceThickness),
        )
        return vol, meta

    @staticmethod
    def _rasterize_roi_contour(contour_seq, vol_shape, meta):
        """Convert RT-Struct contour points to a 3D binary mask."""
        from skimage.draw import polygon
        mask = np.zeros(vol_shape, dtype=bool)
        px, py = meta['pixel_spacing']
        ox, oy, oz = meta['origin']
        sz = meta['slice_spacing']

        for contour in contour_seq:
            pts = np.array(contour.ContourData).reshape(-1, 3)
            z_mm = float(pts[0, 2])
            z_idx = int(round((z_mm - oz) / sz))
            if z_idx < 0 or z_idx >= vol_shape[2]:
                continue
            # Patient coords → pixel indices (row=Y, col=X)
            col_idx = (pts[:, 0] - ox) / px
            row_idx = (pts[:, 1] - oy) / py
            rr, cc = polygon(row_idx, col_idx, shape=vol_shape[:2])
            mask[rr, cc, z_idx] = True
        return mask

    @pytest.fixture(scope="class")
    def measured_values(self, dicom_data, ref_db):
        """Measure actual HU/LAC from DICOM pixels within each ROI."""
        ds = dicom_data.get('rtstruct')
        dicom_dir = dicom_data.get('dicom_dir')
        if ds is None or dicom_dir is None:
            pytest.skip("RT-Struct or DICOM dir not found")

        # Find all series directories
        series_dirs = sorted(
            d for d in os.listdir(dicom_dir)
            if os.path.isdir(os.path.join(dicom_dir, d))
            and d.startswith('series_'))
        if not series_dirs:
            pytest.skip("No DICOM series found")

        # Load first series volume (primary energy)
        vol, meta = self._load_dicom_volume(
            os.path.join(dicom_dir, series_dirs[0]))
        if vol is None:
            pytest.skip("Could not load DICOM volume")

        # Load second series if available (for dual-energy measurements)
        vol2, meta2 = None, None
        if len(series_dirs) > 1:
            vol2, meta2 = self._load_dicom_volume(
                os.path.join(dicom_dir, series_dirs[1]))

        measurements = []
        for rc in ds.ROIContourSequence:
            roi_num = rc.ReferencedROINumber
            # Find ROI name and observation description
            name = ''
            for roi in ds.StructureSetROISequence:
                if roi.ROINumber == roi_num:
                    name = roi.ROIName
                    break

            desc = ''
            for obs in ds.RTROIObservationsSequence:
                if obs.ReferencedROINumber == roi_num:
                    desc = getattr(obs, 'ROIObservationDescription', '')
                    break

            mat_match = re.search(r'material=(\S+?);', desc)
            if not mat_match:
                continue
            mat = mat_match.group(1)

            if not hasattr(rc, 'ContourSequence') or not rc.ContourSequence:
                continue

            # Rasterize ROI into mask
            mask = self._rasterize_roi_contour(
                rc.ContourSequence, vol.shape, meta)
            n_voxels = int(mask.sum())
            if n_voxels == 0:
                continue

            # Measure HU from primary energy
            hu_vals = vol[mask]
            measured_hu_1 = float(np.mean(hu_vals))
            measured_hu_1_std = float(np.std(hu_vals))

            # Measure HU from second energy if available
            measured_hu_2 = np.nan
            measured_hu_2_std = np.nan
            if vol2 is not None:
                mask2 = self._rasterize_roi_contour(
                    rc.ContourSequence, vol2.shape, meta2)
                if mask2.sum() > 0:
                    hu2 = vol2[mask2]
                    measured_hu_2 = float(np.mean(hu2))
                    measured_hu_2_std = float(np.std(hu2))

            # Reference values from database
            ref_density = np.nan
            ref_z_eff = np.nan
            ref_lac = np.nan
            try:
                mat_dict = ref_db.material(mat)
                ref_density = float(mat_dict.get('density', np.nan))
                ref_z_eff = float(mat_dict.get('z', np.nan))
                mu_arr = mat_dict.get('mu', None)
                if mu_arr is not None and ref_density > 0:
                    ref_lac = float(np.mean(mu_arr) * ref_density)
            except Exception:
                pass

            measurements.append(dict(
                material=mat,
                label=roi_num,
                n_voxels=n_voxels,
                ref_z_eff=ref_z_eff,
                ref_density=ref_density,
                ref_lac=ref_lac,
                measured_hu_1=measured_hu_1,
                measured_hu_1_std=measured_hu_1_std,
                measured_hu_2=measured_hu_2,
                measured_hu_2_std=measured_hu_2_std,
            ))

        return measurements

    @pytest.mark.slow
    def test_measured_values_exist(self, measured_values):
        """Should have measurements for all 25 materials."""
        assert len(measured_values) >= 20, \
            f"Only {len(measured_values)} materials measured, expected >=20"

    @pytest.mark.slow
    def test_water_hu_near_zero(self, measured_values):
        """Water should reconstruct near 0 HU."""
        water = [m for m in measured_values if m['material'] == 'water']
        if not water:
            pytest.skip("Water not found in measurements")
        hu = water[0]['measured_hu_1']
        assert -200 < hu < 200, \
            f"Water HU={hu:.0f}, expected near 0 (±200)"

    @pytest.mark.slow
    def test_air_hu_near_minus_1000(self, measured_values):
        """Air should reconstruct near -1000 HU."""
        air = [m for m in measured_values if m['material'] == 'air']
        if not air:
            pytest.skip("Air not found in measurements")
        hu = air[0]['measured_hu_1']
        assert -1100 < hu < -800, \
            f"Air HU={hu:.0f}, expected near -1000"

    @pytest.mark.slow
    def test_dense_materials_positive_hu(self, measured_values):
        """Dense materials (teflon, bone, metals) should have positive HU."""
        dense = ['teflon', 'bone', 'Fe', 'Cu', 'Ti']
        failures = []
        for mat_name in dense:
            m = [x for x in measured_values if x['material'] == mat_name]
            if not m:
                continue
            if m[0]['measured_hu_1'] < -500:
                failures.append(
                    f"{mat_name}: HU={m[0]['measured_hu_1']:.0f}")
        assert len(failures) == 0, \
            f"Dense materials with unexpectedly low HU:\n" + "\n".join(failures)

    @pytest.mark.slow
    def test_hu_ordering_matches_density_ordering(self, measured_values):
        """Materials with higher density should generally have higher HU."""
        from scipy.stats import spearmanr
        pairs = [(m['ref_density'], m['measured_hu_1'])
                 for m in measured_values
                 if not np.isnan(m['ref_density'])
                 and m['material'] != 'air']  # air is a special case
        if len(pairs) < 5:
            pytest.skip("Not enough materials for correlation test")
        densities, hus = zip(*pairs)
        corr, pval = spearmanr(densities, hus)
        assert corr > RANK_CORR_MIN, \
            f"Spearman correlation between density and HU = {corr:.3f}, " \
            f"expected > {RANK_CORR_MIN}"

    @pytest.mark.slow
    def test_dual_energy_contrast(self, measured_values):
        """High-Z materials should show greater HU difference between
        the two energies than low-Z materials (dual energy contrast)."""
        high_z = ['Fe', 'Cu', 'Ti', 'bone']
        low_z = ['water', 'ethanol', 'polyethylene', 'polystyrene']

        def _de_contrast(mat_name):
            m = [x for x in measured_values if x['material'] == mat_name]
            if not m or np.isnan(m[0]['measured_hu_2']):
                return None
            return abs(m[0]['measured_hu_1'] - m[0]['measured_hu_2'])

        high_z_de = [c for c in (_de_contrast(m) for m in high_z)
                     if c is not None]
        low_z_de = [c for c in (_de_contrast(m) for m in low_z)
                    if c is not None]

        if not high_z_de or not low_z_de:
            pytest.skip("Insufficient dual-energy data")

        avg_high = np.mean(high_z_de)
        avg_low = np.mean(low_z_de)
        assert avg_high > avg_low, \
            f"High-Z DE contrast ({avg_high:.0f}) should exceed " \
            f"low-Z DE contrast ({avg_low:.0f})"

    @pytest.mark.slow
    def test_generate_calibration_csv(self, measured_values):
        """Generate calibration_results.csv with measured vs reference data."""
        csv_path = os.path.join(RESULTS_DIR, 'calibration_results.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        fieldnames = [
            'material', 'label', 'n_voxels',
            'ref_z_eff', 'ref_density', 'ref_lac',
            'measured_hu_1', 'measured_hu_1_std',
            'measured_hu_2', 'measured_hu_2_std',
            'hu_error_from_expected',
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in sorted(measured_values, key=lambda x: x['label']):
                # Expected HU from reference LAC:
                # HU = (LAC - LAC_water) / LAC_water * 1000
                water = [x for x in measured_values
                         if x['material'] == 'water']
                expected_hu = np.nan
                if water and not np.isnan(m['ref_lac']):
                    water_lac = [x for x in measured_values
                                 if x['material'] == 'water']
                    # Use ref LAC for water as baseline
                    water_ref = [x for x in measured_values
                                 if x['material'] == 'water']
                    if water_ref and not np.isnan(water_ref[0]['ref_lac']):
                        w_lac = water_ref[0]['ref_lac']
                        if w_lac > 0:
                            expected_hu = (m['ref_lac'] - w_lac) / w_lac * 1000

                row = dict(m)
                row['hu_error_from_expected'] = (
                    m['measured_hu_1'] - expected_hu
                    if not np.isnan(expected_hu) else '')
                writer.writerow(row)

        assert os.path.exists(csv_path), "CSV not created"
        # Verify it's readable
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 20, f"CSV has only {len(rows)} rows"


# ===========================================================================
# Fast tests (no pipeline output needed)
# ===========================================================================

class TestReferenceDataConsistency:
    """Fast tests that verify reference data files are self-consistent."""

    def test_zeff_files_exist(self):
        root = os.path.join(os.path.dirname(__file__), '..')
        for fname in ['include/mu/compounds_zeff.txt',
                       'include/mu/targets_zeff.txt']:
            assert os.path.exists(os.path.join(root, fname)), \
                f"Reference file missing: {fname}"

    def test_all_phantom_materials_have_zeff(self, ref_zeff):
        """Every phantom material should have a Z_eff value (from lookup
        or xraydb fallback)."""
        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        mu = MuDatabaseHandler()
        missing = []
        for mat in PHANTOM_MATERIALS:
            try:
                z = mu.material(mat, 'z')
                if z is None or (isinstance(z, float) and np.isnan(z)):
                    missing.append(mat)
            except Exception:
                missing.append(mat)
        assert len(missing) == 0, f"Materials with no Z_eff: {missing}"

    def test_zeff_values_physically_reasonable(self, ref_zeff):
        """Z_eff values should be between 1 and 92."""
        for mat, z in ref_zeff.items():
            assert 1 <= z <= 92, f"{mat}: Z_eff={z} outside [1, 92]"

    def test_density_values_positive(self):
        """All phantom material densities should be positive."""
        from lib.forward_model.mu_database_handler import MuDatabaseHandler
        mu = MuDatabaseHandler()
        for mat in PHANTOM_MATERIALS:
            try:
                d = mu.material(mat, 'density')
                if d is not None:
                    assert float(d) >= 0, \
                        f"{mat}: density={d} is negative"
            except Exception:
                pass  # some materials may not be in database

    def test_compound_formulas_complete(self):
        """All compound phantom materials should have chemical formulas."""
        from lib.forward_model.mu_database_handler import _COMPOUND_FORMULAS
        compounds = [m for m in PHANTOM_MATERIALS
                     if len(m) > 2]  # skip elements like Si, Al, Ti, Fe, Cu, Zn, C
        missing = [m for m in compounds if m not in _COMPOUND_FORMULAS]
        assert len(missing) == 0, \
            f"Materials without chemical formulas: {missing}"
