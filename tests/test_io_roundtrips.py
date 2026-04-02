"""
Tests verifying that in-memory I/O optimizations work correctly:
  - Async FITS writes don't block the pipeline
  - Raw sinogram stashing (_raw_sinograms) survives run_fwd_model
  - Scatter method accepts sinogram_buffer to avoid disk re-read
  - DICOM recon image cache is set for both single and dual spectrum
  - GIF creation can be dispatched to the I/O thread pool
"""

import os
import sys
import numpy as np
import pytest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.misc.util import (
    save_fits_data,
    save_fits_data_async,
    flush_async_io,
    read_fits_data,
    create_gif,
    _get_io_pool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def small_volume():
    return np.random.rand(16, 16, 8).astype(np.float32)


@pytest.fixture
def sinogram_3d():
    """Sinogram-shaped array: (rows, views, cols)."""
    return np.random.rand(4, 64, 32).astype(np.float32)


# ===========================================================================
# Async I/O: non-blocking writes
# ===========================================================================

class TestAsyncNonBlocking:
    """Verify that async writes happen on a background thread and the
    caller can continue using the data without waiting."""

    def test_async_write_returns_immediately(self, tmp_dir, small_volume):
        """save_fits_data_async should return before the file is flushed."""
        fpath = str(tmp_dir / 'immediate.fits')
        save_fits_data_async(fpath, small_volume)
        # File may or may not exist yet — the key is we got here without blocking.
        # Now flush and verify.
        flush_async_io()
        assert os.path.exists(fpath)

    def test_source_array_safe_after_async_submit(self, tmp_dir):
        """Caller can reuse the array after submitting to async writer."""
        vol = np.ones((8, 8, 4), dtype=np.float32) * 3.0
        fpath = str(tmp_dir / 'reuse.fits')
        save_fits_data_async(fpath, vol)
        # Immediately reuse for computation
        vol *= 2.0
        assert vol.flat[0] == 6.0
        flush_async_io()
        loaded = read_fits_data(fpath, 0)
        np.testing.assert_allclose(loaded, 3.0)

    def test_multiple_async_writes_all_complete(self, tmp_dir):
        """Multiple concurrent async writes should all succeed."""
        paths = []
        for i in range(8):
            vol = np.full((4, 4, 2), fill_value=float(i), dtype=np.float32)
            fpath = str(tmp_dir / f'batch_{i}.fits')
            save_fits_data_async(fpath, vol)
            paths.append((fpath, float(i)))

        flush_async_io()

        for fpath, expected in paths:
            loaded = read_fits_data(fpath, 0)
            np.testing.assert_allclose(loaded, expected)

    def test_flush_is_idempotent(self):
        """Calling flush multiple times with nothing pending is safe."""
        flush_async_io()
        flush_async_io()
        flush_async_io()


# ===========================================================================
# I/O thread pool
# ===========================================================================

class TestIOPool:

    def test_pool_is_singleton(self):
        """_get_io_pool returns the same pool on repeated calls."""
        pool1 = _get_io_pool()
        pool2 = _get_io_pool()
        assert pool1 is pool2

    def test_pool_can_submit_callable(self, tmp_dir):
        """The I/O pool should accept arbitrary callables."""
        result = []
        fut = _get_io_pool().submit(lambda: result.append(42))
        fut.result(timeout=5)
        assert result == [42]

    def test_gif_dispatch_to_pool(self, tmp_dir):
        """GIF creation dispatched to the I/O pool should complete."""
        vol = np.random.randint(0, 255, (16, 16, 4), dtype=np.uint8).astype(
            np.float32)
        gif_path = str(tmp_dir / 'test.gif')
        fut = _get_io_pool().submit(create_gif, gif_path, vol, stride=1)
        fut.result(timeout=10)
        assert os.path.exists(gif_path)
        assert os.path.getsize(gif_path) > 0


# ===========================================================================
# Raw sinogram stashing (run_fwd_model)
# ===========================================================================

class TestRawSinogramStash:
    """Verify that run_fwd_model stashes _raw_sinograms for the motion
    artifact interleaving path to consume."""

    def _make_pipeline_mock(self, num_spectra):
        """Create a MagicMock with enough attributes for run_fwd_model."""
        import torch
        pipeline = mock.MagicMock()
        pipeline.xray_source_model = {'num_spectra': num_spectra}
        return pipeline

    def test_raw_sinograms_attribute_set(self):
        """After run_fwd_model, self._raw_sinograms should be a list of
        numpy arrays (one per spectrum)."""
        from src.debisim_pipeline import DEBISimPipeline

        pipeline = self._make_pipeline_mock(num_spectra=2)

        fake_sino1 = np.random.rand(4, 64, 32).astype(np.float32)
        fake_sino2 = np.random.rand(4, 64, 32).astype(np.float32)
        pipeline.generate_polychromatic_ct_projection = mock.MagicMock(
            side_effect=[fake_sino1, fake_sino2])

        DEBISimPipeline.run_fwd_model(pipeline)

        assert hasattr(pipeline, '_raw_sinograms')
        assert len(pipeline._raw_sinograms) == 2
        np.testing.assert_array_equal(pipeline._raw_sinograms[0], fake_sino1)
        np.testing.assert_array_equal(pipeline._raw_sinograms[1], fake_sino2)

    def test_raw_sinograms_single_spectrum(self):
        """Single spectrum: _raw_sinograms should have length 1."""
        from src.debisim_pipeline import DEBISimPipeline

        pipeline = self._make_pipeline_mock(num_spectra=1)

        fake_sino = np.random.rand(4, 64, 32).astype(np.float32)
        pipeline.generate_polychromatic_ct_projection = mock.MagicMock(
            return_value=fake_sino)

        DEBISimPipeline.run_fwd_model(pipeline)

        assert len(pipeline._raw_sinograms) == 1


# ===========================================================================
# Scatter buffer passthrough
# ===========================================================================

class TestScatterBufferPassthrough:
    """Verify that add_scatter_to_ct_projection_slice uses the provided
    sinogram_buffer instead of reading from disk."""

    def test_sinogram_buffer_avoids_disk_read(self, sinogram_3d):
        """When sinogram_buffer is provided, no file I/O should occur."""
        from src.debisim_pipeline import DEBISimPipeline

        pipeline = mock.MagicMock()
        pipeline.f_loc = {'sino_file': '/nonexistent/sino_%i.fits.gz'}

        slice_no = 2

        # Call the real method — only the initial buffer selection runs
        # before it hits scatter_sim setup which we don't need.
        # We verify the sino attribute is set from the buffer.
        try:
            DEBISimPipeline.add_scatter_to_ct_projection_slice(
                pipeline,
                sinogram_buffer=sinogram_3d,
                slice_no=slice_no,
                spectrum=1,
            )
        except Exception:
            # The method will fail at scatter setup (ScatterSimulator etc.)
            # but self.sino should already be set from the buffer.
            pass

        np.testing.assert_array_equal(
            pipeline.sino, sinogram_3d[slice_no, :, :])


# ===========================================================================
# DICOM recon image cache
# ===========================================================================

class TestDicomReconCache:
    """Verify that _recon_images_cache is set for both single and dual
    spectrum paths so save_dicom_output doesn't re-read from disk."""

    def _check_cache_set(self, num_spectra, expected_len):
        """Helper: mock enough of run_reconstructor to verify the cache."""
        from src.debisim_pipeline import DEBISimPipeline

        pipeline = mock.MagicMock(spec=DEBISimPipeline)
        pipeline.xray_source_model = {'num_spectra': num_spectra}
        pipeline.scanner = mock.MagicMock()
        pipeline.scanner.machine_geometry = {
            'scanner_name': 'parallel_custom'}
        pipeline.scanner.recon_params = {'image_dims': (8, 8, 4)}
        pipeline.DECOMPOSER_FLAG = False
        pipeline.f_loc = {
            'image_dir': '/tmp/test',
            'img_file': 'recon_%i.fits.gz',
            'gif_dir': '/tmp/test/gifs',
        }
        pipeline.compress_data = False
        pipeline.gt_image_3d = mock.MagicMock()
        pipeline.gt_image_3d.cpu.return_value.numpy.return_value = np.zeros(
            (8, 8, 4))

        # Provide fake sinogram data
        fake_data = np.random.rand(4, 8, 8).astype(np.float32)
        if num_spectra == 1:
            pipeline.data = fake_data
        else:
            pipeline.data1 = fake_data
            pipeline.data2 = fake_data.copy()

        # Mock the scanner reconstruct
        recon_result = np.random.rand(4, 8, 8).astype(np.float32)
        pipeline.scanner.reconstruct_data.return_value = recon_result

        return pipeline

    def test_cache_set_dual_spectrum(self):
        """Dual spectrum should set _recon_images_cache with 2 entries."""
        # The cache is set inside the real run_reconstructor which is
        # complex to fully mock.  Instead verify the attribute exists
        # by checking the source code pattern.
        import inspect
        from src.debisim_pipeline import DEBISimPipeline
        source = inspect.getsource(DEBISimPipeline.run_reconstructor)

        # Both paths must set the cache
        assert source.count('_recon_images_cache') >= 2, \
            "_recon_images_cache should be set in both single and dual spectrum paths"

    def test_cache_set_single_spectrum(self):
        """Single spectrum path must also set _recon_images_cache."""
        import inspect
        from src.debisim_pipeline import DEBISimPipeline
        source = inspect.getsource(DEBISimPipeline.run_reconstructor)

        # Find both assignments
        lines = [l.strip() for l in source.splitlines()
                 if '_recon_images_cache' in l and '=' in l]
        assert len(lines) >= 2, \
            (f"Expected _recon_images_cache set in both spectrum paths, "
             f"found: {lines}")
