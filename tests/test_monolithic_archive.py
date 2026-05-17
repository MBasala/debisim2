"""
Tests for MonolithicArchive — RAM-aware streaming archive writer with
parallel serialization, auto-selecting zstd/pigz/gzip.

Verifies:
  - FITS, NPZ, pickle data survive the archive round-trip
  - Writer thread (and serializer thread pool) start and stop cleanly
  - RAM pressure triggers flush without deadlock
  - Partial archives are valid tar files
  - Archive contains expected file entries
"""

import io
import os
import sys
import tarfile
import pickle

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.misc.util import MonolithicArchive


# ---------------------------------------------------------------------------
# Helpers — open the archive the writer actually produced, regardless of
# which compressor was chosen (zstd / pigz / gzip).
# ---------------------------------------------------------------------------

def _open_archive(path):
    """Return a tarfile open in read mode, handling zstd, gzip, or plain tar."""
    if path.endswith('.zst'):
        import zstandard
        fh = open(path, 'rb')
        dctx = zstandard.ZstdDecompressor()
        stream = dctx.stream_reader(fh)
        # Wrap in tarfile via a non-seekable file-like
        return tarfile.open(fileobj=stream, mode='r|')
    elif path.endswith('.gz'):
        return tarfile.open(path, mode='r:gz')
    else:
        return tarfile.open(path, mode='r')


def _read_member(path, arcname):
    """Read a single named member's bytes from an archive of any format."""
    with _open_archive(path) as tar:
        for member in tar:
            if member.name == arcname:
                f = tar.extractfile(member)
                return f.read()
    raise KeyError(arcname)


def _list_members(path):
    """List all member names in an archive of any format."""
    with _open_archive(path) as tar:
        return [m.name for m in tar]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def archive_path(tmp_path):
    return str(tmp_path / 'test_output.tar.gz')


@pytest.fixture
def small_volume():
    return np.random.rand(8, 8, 4).astype(np.float32)


# ===========================================================================
# Basic lifecycle
# ===========================================================================

class TestArchiveLifecycle:

    def test_creates_archive_file(self, archive_path, small_volume):
        ar = MonolithicArchive(archive_path)
        ar.add_fits('test.fits', small_volume)
        ar.close()
        # Writer may pick zstd / pigz / gzip — use the actual path
        assert os.path.exists(ar.path)
        assert os.path.getsize(ar.path) > 0

    def test_close_is_idempotent(self, archive_path):
        ar = MonolithicArchive(archive_path)
        ar.close()
        # Second close should not raise
        ar.close()

    def test_empty_archive_is_valid(self, archive_path):
        ar = MonolithicArchive(archive_path)
        ar.close()
        assert os.path.exists(ar.path)
        assert _list_members(ar.path) == []

    def test_writer_thread_terminates(self, archive_path):
        ar = MonolithicArchive(archive_path)
        assert ar._writer.is_alive()
        ar.close()
        assert not ar._writer.is_alive()


# ===========================================================================
# FITS round-trip
# ===========================================================================

class TestFitsRoundTrip:

    def test_fits_data_survives(self, archive_path, small_volume):
        ar = MonolithicArchive(archive_path)
        ar.add_fits('data/test.fits', small_volume)
        ar.close()

        assert 'data/test.fits' in _list_members(ar.path)
        from astropy.io import fits
        raw = _read_member(ar.path, 'data/test.fits')
        loaded = fits.open(io.BytesIO(raw))[0].data
        np.testing.assert_allclose(loaded, small_volume, rtol=1e-5)

    def test_multiple_fits_files(self, archive_path):
        ar = MonolithicArchive(archive_path)
        vols = {}
        for i in range(5):
            v = np.random.rand(4, 4, 2).astype(np.float32)
            name = f'images/recon_{i}.fits'
            ar.add_fits(name, v)
            vols[name] = v
        ar.close()

        names = _list_members(ar.path)
        assert len(names) == 5
        from astropy.io import fits
        for name, expected in vols.items():
            raw = _read_member(ar.path, name)
            loaded = fits.open(io.BytesIO(raw))[0].data
            np.testing.assert_allclose(loaded, expected, rtol=1e-5)

    def test_compressed_fits(self, archive_path, small_volume):
        """compress=True is honored (CompImageHDU instead of PrimaryHDU)."""
        ar = MonolithicArchive(archive_path)
        ar.add_fits('compressed.fits', small_volume, compress=True)
        ar.close()

        from astropy.io import fits
        raw = _read_member(ar.path, 'compressed.fits')
        hdul = fits.open(io.BytesIO(raw))
        loaded = hdul[1].data  # CompImageHDU is at index 1
        np.testing.assert_allclose(loaded, small_volume, rtol=1e-5)


# ===========================================================================
# NPZ round-trip
# ===========================================================================

class TestNpzRoundTrip:

    def test_npz_data_survives(self, archive_path):
        arr1 = np.random.rand(10, 10).astype(np.float32)
        arr2 = np.arange(20, dtype=np.int32)

        ar = MonolithicArchive(archive_path)
        ar.add_npz('projections/sino_c.npz', compton=arr1, pe=arr2)
        ar.close()

        raw = _read_member(ar.path, 'projections/sino_c.npz')
        data = np.load(io.BytesIO(raw))
        np.testing.assert_allclose(data['compton'], arr1, rtol=1e-5)
        np.testing.assert_array_equal(data['pe'], arr2)


# ===========================================================================
# Pickle round-trip
# ===========================================================================

class TestPickleRoundTrip:

    def test_pickle_survives(self, archive_path):
        obj = {'label': 4, 'material': 'water', 'nested': [1, 2, 3]}
        ar = MonolithicArchive(archive_path)
        ar.add_pickle('metadata.pyc', obj)
        ar.close()

        raw = _read_member(ar.path, 'metadata.pyc')
        loaded = pickle.loads(raw)
        assert loaded == obj


# ===========================================================================
# Mixed types
# ===========================================================================

class TestMixedTypes:

    def test_all_types_in_one_archive(self, archive_path, small_volume):
        ar = MonolithicArchive(archive_path)
        ar.add_fits('images/recon.fits', small_volume)
        ar.add_npz('projections/sino.npz', data=small_volume[:, :, 0])
        ar.add_pickle('meta.pyc', {'key': 'value'})
        ar.add_raw('readme.txt', b'DEBISim2 output archive')
        ar.close()

        names = _list_members(ar.path)
        assert len(names) == 4
        assert 'images/recon.fits' in names
        assert 'projections/sino.npz' in names
        assert 'meta.pyc' in names
        assert 'readme.txt' in names


# ===========================================================================
# Large data (RAM pressure simulation)
# ===========================================================================

class TestRAMPressure:

    def test_large_volume_survives(self, archive_path):
        """A moderately large volume should make it through without error."""
        vol = np.random.rand(64, 64, 32).astype(np.float32)
        ar = MonolithicArchive(archive_path, ram_limit_fraction=0.0001)
        # ram_limit very low → forces immediate flush on every add
        ar.add_fits('large.fits', vol)
        ar.close()

        from astropy.io import fits
        raw = _read_member(ar.path, 'large.fits')
        loaded = fits.open(io.BytesIO(raw))[0].data
        np.testing.assert_allclose(loaded, vol, rtol=1e-5)

    def test_many_files_under_pressure(self, archive_path):
        """Multiple adds with aggressive flush threshold."""
        ar = MonolithicArchive(archive_path, ram_limit_fraction=0.0001)
        for i in range(20):
            v = np.random.rand(8, 8, 4).astype(np.float32)
            ar.add_fits(f'vol_{i:03d}.fits', v)
        ar.close()

        assert len(_list_members(ar.path)) == 20
