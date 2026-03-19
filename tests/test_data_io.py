"""
Tests for FITS and DICOM data I/O functions in lib/misc/util.py.
"""

import os
import numpy as np
import pytest

from lib.misc.util import (
    save_fits_data,
    read_fits_data,
    save_fits_data_async,
    flush_async_io,
    save_dicom_series,
    create_rtstruct,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def small_volume():
    """A small 3D volume for fast tests."""
    return np.random.rand(16, 16, 8).astype(np.float32)


@pytest.fixture
def labeled_volume():
    """A 3D volume with integer labels (simulates gt_image_3d)."""
    vol = np.zeros((32, 32, 10), dtype=np.int32)
    # Block 1: label 4
    vol[5:15, 5:15, 3:7] = 4
    # Block 2: label 5
    vol[18:28, 18:28, 3:7] = 5
    return vol


@pytest.fixture
def sample_roi_list():
    """Minimal ROI list matching the labeled_volume fixture."""
    return [
        dict(label=4, name='block_Al', category='phantom_block',
             material='Al', threat=False, mu=0.5, z_eff=13.0, density=2.7),
        dict(label=5, name='block_Fe', category='phantom_block',
             material='Fe', threat=True, mu=1.2, z_eff=26.0, density=7.87),
    ]


# ===========================================================================
# FITS: synchronous read / write
# ===========================================================================

class TestFitsSyncIO:

    def test_write_and_read_uncompressed(self, tmp_dir, small_volume):
        fpath = str(tmp_dir / 'test.fits')
        save_fits_data(fpath, small_volume, compress=False)
        loaded = read_fits_data(fpath, 0)
        np.testing.assert_allclose(loaded, small_volume, rtol=1e-5)

    def test_write_and_read_compressed(self, tmp_dir, small_volume):
        fpath = str(tmp_dir / 'test.fits.gz')
        save_fits_data(fpath, small_volume, compress=True)
        loaded = read_fits_data(fpath, 1)
        np.testing.assert_allclose(loaded, small_volume, rtol=1e-5)

    def test_roundtrip_preserves_shape(self, tmp_dir):
        vol = np.zeros((64, 32, 16), dtype=np.float64)
        fpath = str(tmp_dir / 'shape_test.fits')
        save_fits_data(fpath, vol)
        loaded = read_fits_data(fpath, 0)
        assert loaded.shape == vol.shape

    def test_roundtrip_preserves_dtype_float32(self, tmp_dir):
        vol = np.ones((8, 8, 4), dtype=np.float32) * 42.0
        fpath = str(tmp_dir / 'f32.fits')
        save_fits_data(fpath, vol)
        loaded = read_fits_data(fpath, 0)
        np.testing.assert_allclose(loaded, vol)

    def test_roundtrip_preserves_negative_values(self, tmp_dir):
        vol = np.array([[[-1.5, 0.0, 2.3]]], dtype=np.float32)
        fpath = str(tmp_dir / 'neg.fits')
        save_fits_data(fpath, vol)
        loaded = read_fits_data(fpath, 0)
        np.testing.assert_allclose(loaded, vol)

    def test_read_nonexistent_file_raises(self, tmp_dir):
        with pytest.raises(Exception):
            read_fits_data(str(tmp_dir / 'does_not_exist.fits'), 0)

    def test_overwrite_existing_file(self, tmp_dir, small_volume):
        fpath = str(tmp_dir / 'overwrite.fits')
        save_fits_data(fpath, small_volume)
        new_vol = small_volume * 2.0
        save_fits_data(fpath, new_vol)
        loaded = read_fits_data(fpath, 0)
        np.testing.assert_allclose(loaded, new_vol, rtol=1e-5)

    def test_large_volume(self, tmp_dir):
        """Verify a moderately large volume survives the round-trip."""
        vol = np.random.rand(128, 128, 64).astype(np.float32)
        fpath = str(tmp_dir / 'large.fits')
        save_fits_data(fpath, vol)
        loaded = read_fits_data(fpath, 0)
        np.testing.assert_allclose(loaded, vol, rtol=1e-5)


# ===========================================================================
# FITS: async I/O
# ===========================================================================

class TestFitsAsyncIO:

    def test_async_write_then_flush_and_read(self, tmp_dir, small_volume):
        fpath = str(tmp_dir / 'async_test.fits')
        save_fits_data_async(fpath, small_volume, compress=False)
        flush_async_io()
        loaded = read_fits_data(fpath, 0)
        np.testing.assert_allclose(loaded, small_volume, rtol=1e-5)

    def test_async_write_compressed(self, tmp_dir, small_volume):
        fpath = str(tmp_dir / 'async_gz.fits.gz')
        save_fits_data_async(fpath, small_volume, compress=True)
        flush_async_io()
        loaded = read_fits_data(fpath, 1)
        np.testing.assert_allclose(loaded, small_volume, rtol=1e-5)

    def test_multiple_async_writes(self, tmp_dir):
        """Submit several writes, flush once, verify all."""
        volumes = {}
        for i in range(5):
            vol = np.random.rand(8, 8, 4).astype(np.float32)
            fpath = str(tmp_dir / f'multi_{i}.fits')
            save_fits_data_async(fpath, vol)
            volumes[fpath] = vol

        flush_async_io()

        for fpath, expected in volumes.items():
            loaded = read_fits_data(fpath, 0)
            np.testing.assert_allclose(loaded, expected, rtol=1e-5)

    def test_async_does_not_corrupt_on_source_mutation(self, tmp_dir):
        """The async writer copies data, so mutating the source after
        submission should not affect the saved file."""
        vol = np.ones((8, 8, 4), dtype=np.float32) * 7.0
        fpath = str(tmp_dir / 'mutation.fits')
        save_fits_data_async(fpath, vol)
        vol[:] = 999.0  # mutate after submission
        flush_async_io()
        loaded = read_fits_data(fpath, 0)
        np.testing.assert_allclose(loaded, 7.0)

    def test_flush_with_no_pending_is_noop(self):
        """Calling flush when there's nothing pending should not raise."""
        flush_async_io()


# ===========================================================================
# FITS: GT label round-trip (integer labels survive float32 storage)
# ===========================================================================

class TestFitsLabelRoundTrip:
    """FITS stores data as float32.  GT label images contain integer labels
    that downstream code (regionprops, torch.where) requires to be integer.
    These tests verify the save→read→cast pattern works correctly."""

    def test_integer_labels_survive_roundtrip(self, tmp_dir):
        """Integer labels saved as float32 should be recoverable via .astype(int)."""
        labels = np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.int32)
        fpath = str(tmp_dir / 'gt_labels.fits')
        save_fits_data(fpath, labels.astype(np.float32))
        loaded = read_fits_data(fpath, 0)
        recovered = loaded.astype(np.int32)
        np.testing.assert_array_equal(recovered, labels)

    def test_fits_returns_float_not_int(self, tmp_dir):
        """read_fits_data returns float even for integer-valued data —
        callers must cast explicitly."""
        labels = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
        fpath = str(tmp_dir / 'gt_int.fits')
        save_fits_data(fpath, labels.astype(np.float32))
        loaded = read_fits_data(fpath, 0)
        assert np.issubdtype(loaded.dtype, np.floating)

    def test_regionprops_rejects_float_labels(self):
        """regionprops raises TypeError on float labels — proves the cast
        is necessary and not just defensive."""
        from skimage.measure import regionprops
        float_labels = np.array([[[0, 1], [2, 3]]], dtype=np.float32)
        with pytest.raises(TypeError, match='Non-integer'):
            regionprops(float_labels)

    def test_regionprops_accepts_int_labels(self):
        """regionprops should work after casting to int."""
        from skimage.measure import regionprops
        labels = np.zeros((4, 4, 4), dtype=np.int32)
        labels[1:3, 1:3, 1:3] = 1
        rprops = regionprops(labels)
        assert len(rprops) == 1
        assert rprops[0].label == 1

    def test_large_label_values_survive(self, tmp_dir):
        """Labels up to ~16M should survive float32 round-trip exactly
        (float32 has 23 mantissa bits → exact integers up to 2^24)."""
        labels = np.array([0, 1, 255, 1000, 16_777_216], dtype=np.int32)
        vol = labels.reshape(1, 1, 5).astype(np.float32)
        fpath = str(tmp_dir / 'big_labels.fits')
        save_fits_data(fpath, vol)
        loaded = read_fits_data(fpath, 0)
        recovered = loaded.astype(np.int32)
        np.testing.assert_array_equal(recovered, labels.reshape(1, 1, 5))


# ===========================================================================
# DICOM: series writing
# ===========================================================================

class TestDicomSeries:

    def test_creates_dcm_files(self, tmp_dir, small_volume):
        out_dir = str(tmp_dir / 'dicom_series')
        os.makedirs(out_dir, exist_ok=True)
        uids = save_dicom_series(out_dir, small_volume)
        dcm_files = [f for f in os.listdir(out_dir) if f.endswith('.dcm')]
        # One .dcm per slice (axis 2)
        assert len(dcm_files) == small_volume.shape[2]

    def test_returns_uid_dict(self, tmp_dir, small_volume):
        out_dir = str(tmp_dir / 'dicom_uids')
        os.makedirs(out_dir, exist_ok=True)
        uids = save_dicom_series(out_dir, small_volume)
        assert 'study_uid' in uids
        assert 'series_uid' in uids
        assert 'frame_uid' in uids
        assert 'sop_uids' in uids
        assert len(uids['sop_uids']) == small_volume.shape[2]

    def test_dicom_pixel_data_roundtrip(self, tmp_dir):
        """Verify pixel values survive the DICOM save→load cycle.

        The DICOM writer stores values as signed int16 with
        RescaleIntercept=0 and RescaleSlope=1.  Input values are
        clipped to [-32768, 32767] and cast directly to int16.
        """
        import pydicom
        vol = np.zeros((16, 16, 4), dtype=np.float32)
        vol[5:10, 5:10, :] = 500.0
        out_dir = str(tmp_dir / 'dicom_pixels')
        os.makedirs(out_dir, exist_ok=True)
        save_dicom_series(out_dir, vol)

        dcm_files = sorted(
            f for f in os.listdir(out_dir) if f.endswith('.dcm'))
        ds = pydicom.dcmread(os.path.join(out_dir, dcm_files[0]))
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

        # The block region should have distinctly higher values than
        # the background, and the relative difference should be ~500 HU.
        bg_val = arr[0, 0]
        block_val = arr[7, 7]
        assert block_val - bg_val == pytest.approx(500.0, abs=2.0)

    def test_dicom_metadata_kvp(self, tmp_dir, small_volume):
        """Verify KVP tag is written when scan_metadata is provided."""
        import pydicom
        out_dir = str(tmp_dir / 'dicom_kvp')
        os.makedirs(out_dir, exist_ok=True)
        metadata = dict(kVp=130, dosage=2e5)
        save_dicom_series(out_dir, small_volume, scan_metadata=metadata)

        dcm_files = sorted(
            f for f in os.listdir(out_dir) if f.endswith('.dcm'))
        ds = pydicom.dcmread(os.path.join(out_dir, dcm_files[0]))
        assert hasattr(ds, 'KVP')
        assert float(ds.KVP) == 130.0

    def test_dicom_custom_pixel_spacing(self, tmp_dir, small_volume):
        """Verify pixel spacing is written correctly."""
        import pydicom
        out_dir = str(tmp_dir / 'dicom_spacing')
        os.makedirs(out_dir, exist_ok=True)
        save_dicom_series(out_dir, small_volume,
                          pixel_spacing=(2.0, 2.0), slice_thickness=3.0)

        dcm_files = sorted(
            f for f in os.listdir(out_dir) if f.endswith('.dcm'))
        ds = pydicom.dcmread(os.path.join(out_dir, dcm_files[0]))
        assert list(ds.PixelSpacing) == [2.0, 2.0]
        assert float(ds.SliceThickness) == 3.0

    def test_dicom_no_metadata(self, tmp_dir, small_volume):
        """Series creation works with no scan_metadata (defaults only)."""
        out_dir = str(tmp_dir / 'dicom_nomd')
        os.makedirs(out_dir, exist_ok=True)
        uids = save_dicom_series(out_dir, small_volume,
                                 scan_metadata=None)
        assert len(uids['sop_uids']) == small_volume.shape[2]


# ===========================================================================
# DICOM: RT-Struct
# ===========================================================================

class TestRTStruct:

    def _make_ct_series(self, tmp_dir, labeled_volume):
        """Helper: write a CT series and return UIDs."""
        ct_dir = str(tmp_dir / 'ct_for_rtstruct')
        os.makedirs(ct_dir, exist_ok=True)
        ct_vol = labeled_volume.astype(np.float32) * 100.0
        return save_dicom_series(ct_dir, ct_vol), ct_dir

    def test_rtstruct_file_created(self, tmp_dir, labeled_volume,
                                   sample_roi_list):
        uids, _ = self._make_ct_series(tmp_dir, labeled_volume)
        rt_path = str(tmp_dir / 'rtstruct.dcm')
        create_rtstruct(rt_path, sample_roi_list, labeled_volume, uids)
        assert os.path.exists(rt_path)

    def test_rtstruct_contains_rois(self, tmp_dir, labeled_volume,
                                    sample_roi_list):
        import pydicom
        uids, _ = self._make_ct_series(tmp_dir, labeled_volume)
        rt_path = str(tmp_dir / 'rtstruct_rois.dcm')
        create_rtstruct(rt_path, sample_roi_list, labeled_volume, uids)

        ds = pydicom.dcmread(rt_path)
        # Should have 2 ROIs
        assert len(ds.StructureSetROISequence) == 2

    def test_rtstruct_roi_names(self, tmp_dir, labeled_volume,
                                sample_roi_list):
        import pydicom
        uids, _ = self._make_ct_series(tmp_dir, labeled_volume)
        rt_path = str(tmp_dir / 'rtstruct_names.dcm')
        create_rtstruct(rt_path, sample_roi_list, labeled_volume, uids)

        ds = pydicom.dcmread(rt_path)
        roi_names = [roi.ROIName for roi in ds.StructureSetROISequence]
        assert 'block_Al' in roi_names
        assert 'block_Fe' in roi_names

    def test_rtstruct_empty_roi_list(self, tmp_dir, labeled_volume):
        """RT-Struct with no ROIs should still produce a valid file."""
        uids, _ = self._make_ct_series(tmp_dir, labeled_volume)
        rt_path = str(tmp_dir / 'rtstruct_empty.dcm')
        create_rtstruct(rt_path, [], labeled_volume, uids)
        assert os.path.exists(rt_path)

    def test_rtstruct_modality(self, tmp_dir, labeled_volume,
                               sample_roi_list):
        """RT-Struct DICOM file should have modality 'RTSTRUCT'."""
        import pydicom
        uids, _ = self._make_ct_series(tmp_dir, labeled_volume)
        rt_path = str(tmp_dir / 'rtstruct_mod.dcm')
        create_rtstruct(rt_path, sample_roi_list, labeled_volume, uids)

        ds = pydicom.dcmread(rt_path)
        assert ds.Modality == 'RTSTRUCT'
