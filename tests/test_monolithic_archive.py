"""
Tests for MonolithicArchive — RAM-aware streaming tar.gz writer backed
by a dedicated subprocess.

Verifies:
  - FITS, NPZ, pickle data survive the archive round-trip
  - Writer subprocess starts and stops cleanly
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
        assert os.path.exists(archive_path)
        assert os.path.getsize(archive_path) > 0

    def test_close_is_idempotent(self, archive_path):
        ar = MonolithicArchive(archive_path)
        ar.close()
        # Second close should not raise
        ar.close()

    def test_empty_archive_is_valid(self, archive_path):
        ar = MonolithicArchive(archive_path)
        ar.close()
        assert os.path.exists(archive_path)
        with tarfile.open(archive_path, 'r:gz') as tar:
            assert len(tar.getnames()) == 0

    def test_writer_process_terminates(self, archive_path):
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

        with tarfile.open(archive_path, 'r:gz') as tar:
            names = tar.getnames()
            assert 'data/test.fits' in names

            member = tar.getmember('data/test.fits')
            f = tar.extractfile(member)
            from astropy.io import fits
            hdul = fits.open(io.BytesIO(f.read()))
            loaded = hdul[0].data
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

        with tarfile.open(archive_path, 'r:gz') as tar:
            assert len(tar.getnames()) == 5
            for name, expected in vols.items():
                f = tar.extractfile(tar.getmember(name))
                from astropy.io import fits
                loaded = fits.open(io.BytesIO(f.read()))[0].data
                np.testing.assert_allclose(loaded, expected, rtol=1e-5)

    def test_compressed_fits(self, archive_path, small_volume):
        """CompImageHDU (compress=True) should also survive."""
        ar = MonolithicArchive(archive_path)
        ar.add_fits('compressed.fits', small_volume, compress=True)
        ar.close()

        with tarfile.open(archive_path, 'r:gz') as tar:
            f = tar.extractfile(tar.getmember('compressed.fits'))
            from astropy.io import fits
            hdul = fits.open(io.BytesIO(f.read()))
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

        with tarfile.open(archive_path, 'r:gz') as tar:
            f = tar.extractfile(tar.getmember('projections/sino_c.npz'))
            data = np.load(io.BytesIO(f.read()))
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

        with tarfile.open(archive_path, 'r:gz') as tar:
            f = tar.extractfile(tar.getmember('metadata.pyc'))
            loaded = pickle.loads(f.read())
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

        with tarfile.open(archive_path, 'r:gz') as tar:
            names = tar.getnames()
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

        with tarfile.open(archive_path, 'r:gz') as tar:
            f = tar.extractfile(tar.getmember('large.fits'))
            from astropy.io import fits
            loaded = fits.open(io.BytesIO(f.read()))[0].data
            np.testing.assert_allclose(loaded, vol, rtol=1e-5)

    def test_many_files_under_pressure(self, archive_path):
        """Multiple adds with aggressive flush threshold."""
        ar = MonolithicArchive(archive_path, ram_limit_fraction=0.0001)
        for i in range(20):
            v = np.random.rand(8, 8, 4).astype(np.float32)
            ar.add_fits(f'vol_{i:03d}.fits', v)
        ar.close()

        with tarfile.open(archive_path, 'r:gz') as tar:
            assert len(tar.getnames()) == 20
