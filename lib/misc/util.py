import builtins
import sys
import io
import subprocess
import tarfile
import time as _time
import atexit as _atexit

import numpy as np
from numpy import *  # noqa: F401,F403 - intentional wildcard re-export
from numpy.linalg import eigh

import os, pickle, pydicom, logging, logging.handlers
import scipy.sparse as sp
import scipy.misc as misc
from concurrent.futures import ThreadPoolExecutor
import threading

import psutil

from astropy.io import fits as pyfits
from pydicom import uid
from skimage.measure import regionprops
from skimage.transform import rescale

from lib.__init__ import *
from sys import stdout as stdout
from PIL import Image


class DicomCoordinateMapper:
    """Centralizes coordinate transforms between DEBISim internal volumes
    and DICOM pixel/patient coordinate space.

    DEBISim stores volumes as [X, Y, Z].
    DICOM stores pixel arrays as [row, col] with orientation defined by
    ImageOrientationPatient (IOP).

    With IOP [1,0,0, 0,1,0]:
      - Row direction = patient +X
      - Column direction = patient +Y
      - The viewer affine maps: col → patient X, row → patient Y
      - Therefore pixel_array[row, col] = data[Y_idx, X_idx]
      - A .T transpose converts [X, Y] → [Y, X] for storage

    This class owns these operations:
      1. volume_slice_to_pixels() — for DICOM CT series writing
      2. voxel_to_patient()       — for RT-Struct contour generation

    All transforms are derived from the same IOP and pixel_spacing,
    so they stay consistent automatically.
    """

    # Standard axial orientation: row=+X, col=+Y
    IOP = [1, 0, 0, 0, 1, 0]

    def __init__(self, pixel_spacing=(1.0, 1.0), slice_thickness=1.0,
                 origin=(0.0, 0.0, 0.0)):
        self.pixel_spacing = tuple(float(p) for p in pixel_spacing)
        self.slice_thickness = float(slice_thickness)
        self.origin = tuple(float(o) for o in origin)
        # Row spacing = ps[0], Col spacing = ps[1]
        # With IOP [1,0,0,0,1,0]: col→X uses ps[1], row→Y uses ps[0]
        # but pydicom PixelSpacing is [row_spacing, col_spacing]
        self._ps_row = self.pixel_spacing[0]
        self._ps_col = self.pixel_spacing[1]

    def volume_slice_to_pixels(self, vol_slice_xy):
        """Convert a [X, Y] volume slice to DICOM pixel array [row=Y, col=X].

        Parameters
        ----------
        vol_slice_xy : ndarray, shape (nx, ny)
            A 2D slice from the internal volume (X along axis 0, Y along axis 1).

        Returns
        -------
        pixel_data : ndarray, shape (ny, nx)
            Transposed and clipped to int16 for DICOM storage.
        """
        return np.clip(vol_slice_xy.T, -32768, 32767).astype(np.int16)

    def pixel_rows_cols(self, vol_slice_xy):
        """Return (Rows, Columns) for the DICOM header after transpose."""
        return vol_slice_xy.shape[1], vol_slice_xy.shape[0]  # (ny, nx)

    def voxel_to_patient(self, row_idx, col_idx, z_idx):
        """Convert GT volume indices (X=row_idx, Y=col_idx, Z=z_idx)
        to DICOM patient coordinates (x_mm, y_mm, z_mm).

        The GT volume is indexed as [X, Y, Z].  find_contours on
        gt[:, :, z] returns (row, col) = (X_idx, Y_idx).

        With IOP [1,0,0, 0,1,0], the viewer affine maps:
          patient X = origin_x + col_idx * ps_col   (col tracks X)
          patient Y = origin_y + row_idx * ps_row   (row tracks Y)

        But our contour (row, col) from find_contours on [X, Y] gives
        row=X_idx, col=Y_idx.  So:
          patient X = X_idx * ps_row   (from the first axis)
          patient Y = Y_idx * ps_col   (from the second axis)
        """
        x_mm = float(row_idx) * self._ps_row + self.origin[0]
        y_mm = float(col_idx) * self._ps_col + self.origin[1]
        z_mm = float(z_idx) * self.slice_thickness + self.origin[2]
        return x_mm, y_mm, z_mm

    def image_position_patient(self, z_idx):
        """Return ImagePositionPatient for a given slice index."""
        return [self.origin[0], self.origin[1],
                self.origin[2] + z_idx * self.slice_thickness]

    def dicom_tags(self):
        """Return a dict of orientation-related DICOM tags."""
        return dict(
            ImageOrientationPatient=list(self.IOP),
            PixelSpacing=list(self.pixel_spacing),
            SliceThickness=str(self.slice_thickness),
            SpacingBetweenSlices=str(self.slice_thickness),
        )


class Logger(object):
    """
    ---------------------------------------------------------------------------
    Module Description:
    This module is a tool for directing sys.out to both a file and printing
    in terminal. Example usage:

    sys.stdout = Logger(log_name)

    ---------------------------------------------------------------------------
    """

    def __init__(self, fname):
        """
        -----------------------------------------------------------------------
        Constructor

        :param fname: file name
        -----------------------------------------------------------------------
        """

        self.terminal = sys.stdout
        self.fname = fname

    def write(self, message):
        """
        -----------------------------------------------------------------------
        put message on the log file

        :param message:     message string to add
        :return:
        -----------------------------------------------------------------------
        """

        self.terminal.write(message)
        self.log = open(self.fname, "a")
        self.log.write(message)
        self.log.close()

    def flush(self):
        """
        -----------------------------------------------------------------------
        this flush method is needed for python 3 compatibility.
        this handles the flush command by doing nothing.
        you might want to specify some extra behavior here.

        :return:
        -----------------------------------------------------------------------
        """

        pass
# =============================================================================
# Class Ends
# =============================================================================


def get_logger(lname, logfile):
    """
    ---------------------------------------------------------------------------
    Function to create a python logger for a given class.

    :param lname:       name for logger - printed on terminal
    :param logfile:     log file path
    :return:
    ---------------------------------------------------------------------------
    """

    # Create logger object
    logger = logging.getLogger(lname)
    logger.setLevel(logging.INFO)

    # Create file handler with buffered writes to reduce I/O syscalls.
    # MemoryHandler capacity is a RECORD COUNT (not bytes).
    # 256 records ≈ 50-100KB of typical log output before flush.
    f_handler = logging.FileHandler(logfile, mode='a')
    buffered_f_handler = logging.handlers.MemoryHandler(
        capacity=256,                 # flush every 256 log records
        flushLevel=logging.WARNING,   # flush immediately on warnings/errors
        target=f_handler,
    )

    s_handler = logging.StreamHandler(stdout)

    # Create formatter
    formatter = logging.Formatter(
                        '[%(asctime)s] [%(name)s] %(levelname)s: %(message)s')
    f_handler.setFormatter(formatter)
    s_handler.setFormatter(formatter)

    logger.addHandler(buffered_f_handler)
    logger.addHandler(s_handler)
    logger.propagate = False

    return logger
# -----------------------------------------------------------------------------


def read_fits_data(input_file_name, field=1):
    """
    ---------------------------------------------------------------------------
    Loads a FITS image file

    :param input_file_name - file path
    :return image as a numpy ndarray
    ---------------------------------------------------------------------------
    """

    return  pyfits.open(input_file_name,
                        ignore_missing_end=True)[field].data
# -----------------------------------------------------------------------------


def save_fits_data(file_path, out_image, compress=False):
    """
    ---------------------------------------------------------------------------
    Save an image as a FITS file

    :param file_path:   path to the fits file
    :param out_image:   output image to be saved
    :return:
    ---------------------------------------------------------------------------
    """

    if os.path.exists(file_path):
        os.remove(file_path)

    imheader = pyfits.Header()

    if compress:
        hdu_list = pyfits.CompImageHDU(out_image, imheader)
    else:
        hdu_list = pyfits.PrimaryHDU(out_image, imheader)

    hdu_list.writeto(file_path)
# -----------------------------------------------------------------------------


# ---- Async I/O for FITS saving ----------------------------------------------
# A module-level thread pool for non-blocking FITS writes.  The data is
# copied (numpy array) before submission so the caller can safely mutate
# or delete the original without corrupting the write.

_io_pool = None
_io_futures = []
_io_lock = threading.Lock()


def _get_io_pool():
    """Lazily create the I/O thread pool on first use."""
    global _io_pool
    if _io_pool is None:
        import atexit
        _io_pool = ThreadPoolExecutor(max_workers=2,
                                       thread_name_prefix='fits_io')
        atexit.register(_io_pool.shutdown, wait=True)
    return _io_pool


def submit_async_io(fn, *args, **kwargs):
    """Submit a callable to the I/O thread pool and track its future.

    Completed futures are pruned on each call; any exception from a
    finished write is raised immediately so failures are not silently lost.

    :returns: ``concurrent.futures.Future``
    """
    fut = _get_io_pool().submit(fn, *args, **kwargs)
    with _io_lock:
        surviving = []
        for f in _io_futures:
            if f.done():
                exc = f.exception()
                if exc is not None:
                    raise exc
            else:
                surviving.append(f)
        surviving.append(fut)
        _io_futures[:] = surviving
    return fut


def save_fits_data_async(file_path, out_image, compress=False):
    """Submit a FITS save to the background I/O thread pool.

    The image array is copied before submission so the caller can safely
    reuse or free the memory immediately.

    :param file_path:   path to the fits file
    :param out_image:   numpy array to save
    :param compress:    whether to use FITS compression
    """
    data_copy = out_image.copy()
    submit_async_io(save_fits_data, file_path, data_copy, compress)


def flush_async_io():
    """Block until all pending async I/O has completed.

    Call this before the pipeline exits or before reading back saved files.
    Raises the first exception encountered, if any.
    """
    with _io_lock:
        pending = list(_io_futures)
        _io_futures.clear()

    for fut in pending:
        fut.result()  # raises if the write failed
# -----------------------------------------------------------------------------


# =============================================================================
# Monolithic tar.gz archive — threaded writer with pigz for parallel gzip
# =============================================================================

_ARCHIVE_SENTINEL = None  # sentinel value to signal writer shutdown


def _try_zstd_stream(archive_path, threads):
    """Open a streaming zstd writer if the ``zstandard`` package is
    available.  Returns ``(out_fh, stream_writer)`` pair or ``(None, None)``.

    zstd is 5-15× faster than gzip at similar compression ratios and has
    native multi-thread support — by far the best choice when available.
    """
    try:
        import zstandard
    except ImportError:
        return None, None
    out_fh = None
    try:
        # Re-suffix to .zst so consumers know the format
        if not archive_path.endswith('.zst'):
            archive_path = archive_path[:-3] + '.zst' \
                if archive_path.endswith('.gz') else archive_path + '.zst'
        # zstd level 3 is the sweet spot: ~gzip-6 ratio at ~3-5× speed
        cctx = zstandard.ZstdCompressor(
            level=3, threads=(threads if threads > 0 else -1))
        out_fh = open(archive_path, 'wb')
        stream = cctx.stream_writer(out_fh)
        return (out_fh, stream), archive_path
    except Exception:
        # Same file-handle leak the fix to _try_pigz addressed: if open()
        # succeeded but stream_writer (or anything after) raised, close
        # out_fh so it doesn't stay open and block the fallback compressor
        # from opening the same path on Windows.
        if out_fh is not None:
            try:
                out_fh.close()
            except Exception:
                pass
        return None, None


def _try_pigz(archive_path, threads):
    """Start a pigz subprocess for parallel gzip.  Returns Popen or None.

    On any error (pigz not installed, file open failure, etc.) the
    output file handle is closed so it doesn't leak — important on
    Windows where a stale handle on archive_path would block the
    gzip-fallback path from opening it.
    """
    out_fh = None
    try:
        cmd = ['pigz', '-c']
        if threads > 0:
            cmd += ['-p', str(threads)]
        out_fh = open(archive_path, 'wb')
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out_fh)
        proc._out_fh = out_fh          # prevent GC closing file handle
        return proc
    except (FileNotFoundError, OSError):
        if out_fh is not None:
            try:
                out_fh.close()
            except Exception:
                pass
        return None


def _archive_writer_loop(q, archive_path, threads, written_event,
                         written_count_lock, written_count_list,
                         final_path_holder, error_holder):
    """Writer thread entry point — chooses the fastest available compressor.

    Tries in order: zstd (multi-thread, ~10-20× faster than gzip), pigz
    (multi-thread gzip), then Python tarfile w:gz (single-thread fallback).

    Uses a threading.Thread (not a subprocess) so that data passes
    by reference — no pickling, no pipe buffer limits for multi-GB
    FITS arrays.  zstd/zlib/pigz all release the GIL during compression,
    so GPU work runs concurrently.

    Parameters
    ----------
    q : queue.Queue
        Thread-safe queue of (arcname, data_bytes) tuples.
    archive_path : str
        Requested output path (suffix may be rewritten to .tar.zst).
    threads : int
        Compression threads (0 = auto).
    written_event : threading.Event
        Set after each item is written — used for RAM-pressure drain.
    written_count_lock : threading.Lock
        Protects written_count_list.
    written_count_list : list[int]
        Single-element list holding the write count (mutable container).
    final_path_holder : list[str]
        Single-element list — writer records the actual path used here so
        the caller can find the output (in case the suffix changed).
    error_holder : list
        Single-element list — writer stores any exception here so the
        main thread can surface it on close().  ``written_event`` is
        always set on error to prevent the main thread from deadlocking
        in ``_wait_for_writer_drain``.
    """
    pigz_proc = None
    tar = None
    zstd_pair = None     # (out_fh, stream_writer) when zstd path is taken
    actual_path = archive_path
    try:
        # ----- preferred: zstd ------------------------------------------------
        zstd_result, zstd_path = _try_zstd_stream(archive_path, threads)
        if zstd_result is not None:
            zstd_pair = zstd_result
            actual_path = zstd_path
            tar = tarfile.open(mode='w|', fileobj=zstd_pair[1])
        else:
            # ----- fallback: pigz subprocess ---------------------------------
            pigz_proc = _try_pigz(archive_path, threads)
            if pigz_proc is not None:
                tar = tarfile.open(mode='w|', fileobj=pigz_proc.stdin)
            else:
                # ----- last resort: single-thread Python gzip ----------------
                tar = tarfile.open(archive_path, 'w:gz', compresslevel=6)

        final_path_holder[0] = actual_path

        while True:
            item = q.get()
            if item is _ARCHIVE_SENTINEL:
                break
            arcname, data = item
            info = tarfile.TarInfo(name=arcname)
            info.size = len(data)
            info.mtime = int(_time.time())
            tar.addfile(info, io.BytesIO(data))
            # Update progress (thread-safe via lock)
            with written_count_lock:
                written_count_list[0] += 1
            written_event.set()
    except BaseException as exc:
        # Capture for close() to re-raise.  Always wake the main thread
        # so _wait_for_writer_drain doesn't hang forever.
        error_holder.append(exc)
        written_event.set()
    finally:
        if tar is not None:
            try:
                tar.close()
            except Exception:
                pass
        if zstd_pair is not None:
            try:
                zstd_pair[1].close()   # flush + finalize zstd stream
                zstd_pair[0].close()   # close underlying file
            except Exception:
                pass
        if pigz_proc is not None:
            try:
                pigz_proc.stdin.close()
                pigz_proc.wait(timeout=30)
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Serializer thread pool — moves Python serialization off the main thread.
# Items submitted as (arcname, kind, payload) where:
#   kind='fits'   -> payload = (array, compress_bool)
#   kind='npz'    -> payload = dict of arrays
#   kind='dicom'  -> payload = pydicom Dataset
#   kind='pickle' -> payload = arbitrary picklable object
#   kind='raw'    -> payload = bytes (already serialized)
# Workers convert payload -> bytes and push (arcname, bytes) to the writer
# queue.  Backpressure from the writer queue propagates naturally.
# -----------------------------------------------------------------------------

def _serialize_item(kind, payload):
    """Convert a payload into bytes ready for the writer queue."""
    buf = io.BytesIO()
    if kind == 'raw':
        return payload
    elif kind == 'fits':
        array, compress = payload
        hdu = (pyfits.CompImageHDU(array, pyfits.Header()) if compress
               else pyfits.PrimaryHDU(array, pyfits.Header()))
        hdu.writeto(buf)
    elif kind == 'npz':
        # When destined for an archive, the outer compressor already
        # handles compression — use savez (uncompressed) to avoid wasting
        # CPU compressing twice.
        np.savez(buf, **payload)
    elif kind == 'dicom':
        payload.save_as(buf)
    elif kind == 'pickle':
        pickle.dump(payload, buf, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unknown serializer kind: {kind!r}")
    return buf.getvalue()


def _serializer_loop(in_q, out_q, written_lock, written_count_list,
                     written_event, error_holder, logger=None):
    """Serializer thread entry point.

    Pulls (arcname, kind, payload) from ``in_q``, calls ``_serialize_item``
    (which can take 10-100 ms for big FITS / many-element DICOM), and
    pushes (arcname, bytes) to ``out_q`` for the writer thread.

    Multiple of these can run in parallel — pydicom/numpy/astropy release
    the GIL during their hot inner loops (memcpy, zlib, etc.), so CPU
    cores are actually utilized.

    On serialization failure for any item, the exception is captured in
    ``error_holder`` (surfaced on close()), logged if a logger is
    provided, and the item is dropped — incrementing written_count_list
    so RAM-pressure tracking doesn't deadlock waiting for it.  This keeps
    the pipeline alive while still alerting the caller to data loss.
    """
    while True:
        item = in_q.get()
        if item is _ARCHIVE_SENTINEL:
            # Forward sentinel to writer once all serializers exit
            break
        arcname, kind, payload = item
        try:
            data = _serialize_item(kind, payload)
            out_q.put((arcname, data))
        except BaseException as exc:
            # Record so close() can surface it; never silently lose data
            error_holder.append((arcname, kind, exc))
            if logger is not None:
                try:
                    logger.error(
                        f"archive serializer failed on {arcname!r} "
                        f"(kind={kind}): {exc!r}")
                except Exception:
                    pass
            with written_lock:
                written_count_list[0] += 1
            written_event.set()


class MonolithicArchive:
    """RAM-aware streaming archive writer with parallel serialization.

    Data flow (two-stage pipeline)::

        main thread:    add_fits/npz/dicom/pickle  →  serializer queue
                        (immediate return — no serialization on main)

        serializer pool (N threads):
                        in_q  →  pydicom/astropy/numpy serialize  →  out_q

        writer thread:  out_q  →  tarfile  →  zstd (preferred) /
                                              pigz / gzip  →  disk

    Compressor preference: zstd (10-20× faster than gzip) → pigz (multi-
    thread gzip) → Python tarfile w:gz (single-thread fallback).  Writer
    auto-picks the fastest available without user intervention.

    Serializer pool offloads slow Python serialization (especially
    pydicom which is ~Python-level per DataElement) from the main thread,
    so the compute pipeline isn't blocked.  Workers run in parallel and
    benefit from GIL releases inside the underlying libraries.

    Inner-compression (savez_compressed, CompImageHDU) is intentionally
    bypassed when going into an archive: the outer compressor already
    compresses, so doubling up only wastes CPU.

    Synchronization occurs only when:
      1. Available RAM drops below threshold — main blocks until the
         writer drains enough items.
      2. ``close()`` is called at pipeline end.
    """

    def __init__(self, archive_path, compression_threads=0,
                 ram_limit_fraction=0.5, logger=None,
                 serializer_workers=None):
        self._archive_path = archive_path
        self._ram_limit = ram_limit_fraction
        self._logger = logger

        # Progress tracking (thread-safe)
        self._written_event = threading.Event()
        self._written_lock = threading.Lock()
        self._written_count = [0]   # mutable container shared with writer
        self._enqueued_count = 0
        self._total_enqueued_bytes = 0

        # Writer thread sets final archive path here (may differ from
        # requested if zstd path is chosen: .tar.gz -> .tar.zst)
        self._final_path_holder = [archive_path]

        # Error capture — populated by writer/serializer threads on failure
        # so close() can surface them (rather than letting the main thread
        # think a partial/empty archive was a successful run).
        self._writer_error = []         # writer thread fatal init/write errors
        self._serializer_errors = []    # per-item serializer failures

        # Two-stage pipeline queues
        import queue
        self._serializer_queue = queue.Queue()   # main thread -> serializers
        self._writer_queue = queue.Queue()       # serializers -> writer

        # Number of serializer workers.  Default: half the CPUs, capped
        # to avoid contention with the compute pipeline.
        if serializer_workers is None:
            try:
                cpu = os.cpu_count() or 4
            except Exception:
                cpu = 4
            serializer_workers = builtins.max(2, builtins.min(4, cpu // 2))
        self._n_serializers = serializer_workers

        # Writer thread
        self._writer = threading.Thread(
            target=_archive_writer_loop,
            args=(self._writer_queue, archive_path, compression_threads,
                  self._written_event, self._written_lock,
                  self._written_count, self._final_path_holder,
                  self._writer_error),
            daemon=True,
            name='archive-writer',
        )
        self._writer.start()

        # Serializer pool
        self._serializers = [
            threading.Thread(
                target=_serializer_loop,
                args=(self._serializer_queue, self._writer_queue,
                      self._written_lock, self._written_count,
                      self._written_event, self._serializer_errors,
                      self._logger),
                daemon=True,
                name=f'archive-serializer-{i}',
            )
            for i in range(self._n_serializers)
        ]
        for t in self._serializers:
            t.start()

    # ---- public API: hand off to serializer pool (non-blocking) ---------

    def add_fits(self, arcname, array, compress=False):
        """Submit a numpy array for FITS serialization + archiving.

        ``compress`` defaults to False because the outer zstd/gzip layer
        already compresses; using CompImageHDU on top doubles up CPU
        with no size benefit.

        The array is defensively copied so the caller can safely free
        or mutate it as soon as this method returns — the serializer
        thread may not pick the item up for many milliseconds.

        Uses ``np.array(arr, copy=True, order='C')`` rather than the
        chained ``np.ascontiguousarray(arr).copy()``: for non-contiguous
        inputs the latter would copy twice (once to make it contiguous,
        once for the explicit copy), transiently doubling peak RAM for
        multi-GB volumes.
        """
        array_copy = np.array(array, copy=True, order='C')
        self._submit(arcname, 'fits', (array_copy, compress),
                     estimated_bytes=array_copy.nbytes)

    def add_npz(self, arcname, **arrays):
        """Submit numpy arrays for .npz serialization + archiving.

        Always uses uncompressed savez — the archive layer compresses.
        Arrays are defensively copied (see add_fits for rationale and
        the single-copy idiom).
        """
        arrays_copy = {k: np.array(v, copy=True, order='C')
                       for k, v in arrays.items()}
        est = builtins.sum(a.nbytes for a in arrays_copy.values()) \
            if arrays_copy else 0
        self._submit(arcname, 'npz', arrays_copy, estimated_bytes=est)

    def add_dicom(self, arcname, dataset):
        """Submit a pydicom Dataset for serialization + archiving."""
        # pydicom Dataset doesn't expose nbytes; estimate from PixelData
        est = 0
        try:
            pd = getattr(dataset, 'PixelData', b'')
            est = len(pd) if pd else 4096
        except Exception:
            est = 4096
        self._submit(arcname, 'dicom', dataset, estimated_bytes=est)

    def add_pickle(self, arcname, obj):
        """Submit an object for pickle serialization + archiving."""
        # Conservative estimate; refined when actual bytes are produced
        self._submit(arcname, 'pickle', obj, estimated_bytes=64 * 1024)

    def add_raw(self, arcname, data_bytes):
        """Enqueue pre-serialized bytes (bypasses serializer pool)."""
        # Raw bytes don't need serialization — push straight to writer
        self._writer_queue.put((arcname, data_bytes))
        self._enqueued_count += 1
        self._total_enqueued_bytes += len(data_bytes)
        if self._ram_pressure():
            self._wait_for_writer_drain()

    # ---- internal: submit + backpressure --------------------------------

    def _submit(self, arcname, kind, payload, estimated_bytes=0):
        """Submit work to the serializer pool. Block only if RAM is tight.

        The main thread returns immediately — serialization happens on
        worker threads concurrent with compute work.
        """
        self._serializer_queue.put((arcname, kind, payload))
        self._enqueued_count += 1
        # estimated_bytes is approximate but good enough for RAM pressure
        self._total_enqueued_bytes += estimated_bytes

        if self._ram_pressure():
            self._wait_for_writer_drain()

    def _ram_pressure(self):
        """True if estimated in-flight bytes exceed available RAM limit."""
        avail = psutil.virtual_memory().available
        return self._inflight_bytes_estimate() > avail * self._ram_limit

    def _inflight_bytes_estimate(self):
        """Estimate bytes still in flight (not yet written to disk)."""
        with self._written_lock:
            written = self._written_count[0]
        pending = builtins.max(0, self._enqueued_count - written)
        if self._enqueued_count == 0:
            return 0
        avg_size = self._total_enqueued_bytes / self._enqueued_count
        return avg_size * pending

    def _wait_for_writer_drain(self):
        """Block until the writer has processed enough items to relieve
        RAM pressure.  This is the ONLY sync point during normal operation.

        Bails out early if the writer thread has died (e.g. its
        initialization raised) so the main thread doesn't deadlock
        waiting on an event that will never fire again.
        """
        if self._logger:
            est = self._inflight_bytes_estimate()
            self._logger.info(
                f"MonolithicArchive: RAM pressure — waiting for writer "
                f"(in-flight ~{est / 1e9:.1f} GB)")
        while self._ram_pressure():
            self._written_event.wait(timeout=0.1)
            self._written_event.clear()
            if not self._writer.is_alive():
                # Writer died (likely raised during init); stop waiting so
                # the main thread can reach close() and surface the error.
                break

    # ---- lifecycle ------------------------------------------------------

    @property
    def path(self):
        """Actual on-disk archive path.  May differ from the path passed
        to __init__ if the writer chose a different compressor (e.g.
        zstd renames .tar.gz → .tar.zst)."""
        return self._final_path_holder[0]

    def close(self):
        """Signal serializers, then writer to finish; wait for completion.

        Surfaces any fatal writer error or per-item serializer errors so
        the caller knows the archive may be incomplete.  A fatal writer
        error is raised; per-item serializer failures are logged and
        attached to the raised RuntimeError if the writer was also fatal,
        otherwise logged as a warning.
        """
        # Send one sentinel per serializer; they exit and the writer drains
        # whatever is left in writer_queue.  After all serializers exit, the
        # writer needs its own sentinel.
        for _ in range(self._n_serializers):
            self._serializer_queue.put(_ARCHIVE_SENTINEL)
        for t in self._serializers:
            t.join(timeout=300)
        self._writer_queue.put(_ARCHIVE_SENTINEL)
        self._writer.join(timeout=300)

        # The writer may have written to a different path (e.g. .tar.zst
        # if zstd was available).  Surface the actual path back.
        actual_path = self._final_path_holder[0]
        self._archive_path = actual_path
        if self._logger and os.path.exists(actual_path):
            size_mb = os.path.getsize(actual_path) / 1e6
            self._logger.info(
                f"Archive written: {actual_path} ({size_mb:.1f} MB)")

        # Surface captured errors.  A fatal writer error supersedes any
        # serializer errors (writer dying means the archive is unusable).
        if self._writer_error:
            n_dropped = len(self._serializer_errors)
            raise RuntimeError(
                f"MonolithicArchive writer failed: {self._writer_error[0]!r}"
                + (f"  ({n_dropped} item(s) also dropped by serializer "
                   f"pool: {[e[0] for e in self._serializer_errors[:5]]})"
                   if n_dropped else "")
            ) from self._writer_error[0]

        if self._serializer_errors and self._logger:
            dropped_names = [e[0] for e in self._serializer_errors[:10]]
            self._logger.warning(
                f"MonolithicArchive: {len(self._serializer_errors)} item(s) "
                f"dropped by serializer pool — archive is incomplete.  "
                f"First failed arcnames: {dropped_names}")


# =============================================================================


def save_dicom_series(output_dir, volume_3d, patient_id='DEBISim',
                      study_description='Simulated CT',
                      series_description='Recon', series_number=1,
                      pixel_spacing=(1.0, 1.0), slice_thickness=1.0,
                      scan_metadata=None, archive=None):
    """
    Save a 3D numpy array as a DICOM CT image series (one .dcm per slice).

    :param output_dir:  directory to write .dcm files into (also used as
                        archive prefix when archive is set)
    :param volume_3d:   3D numpy array (H x W x D), values in HU
    :param patient_id:  DICOM PatientID
    :param study_description:  DICOM StudyDescription
    :param series_description: DICOM SeriesDescription
    :param series_number:      DICOM SeriesNumber
    :param pixel_spacing:      (row_spacing, col_spacing) in mm
    :param slice_thickness:    slice thickness in mm
    :param scan_metadata:      dict with scanner/source parameters for DICOM tags:
        scanner_name, geometry, scan_type, source_origin, origin_det,
        det_row_count, det_col_count, sens_spacing_x, sens_spacing_y,
        gantry_diameter, num_views, kVp, dosage, recon_algo,
        image_dims, fov, anode_angle
    :param archive:    optional MonolithicArchive — when set, slices are
                       serialized to bytes and dispatched via archive.add_raw()
                       instead of writing to disk
    :returns: dict with keys study_uid, series_uid, frame_uid, sop_uids
    """
    import datetime
    import copy as _copy
    from pydicom.dataset import Dataset, FileDataset

    if scan_metadata is None:
        scan_metadata = {}

    if archive is None:
        os.makedirs(output_dir, exist_ok=True)

    study_uid = uid.generate_uid()
    series_uid = uid.generate_uid()
    frame_uid = uid.generate_uid()
    # Generate all SOP UIDs up front; cheaper than one per loop iteration
    num_slices = volume_3d.shape[2]
    sop_uids = [uid.generate_uid() for _ in range(num_slices)]

    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d')
    time_str = now.strftime('%H%M%S.%f')

    # Derive DICOM fields from scan metadata
    kvp = scan_metadata.get('kVp', '')
    scanner_name = scan_metadata.get('scanner_name', 'DEBISim2')
    geometry = scan_metadata.get('geometry', 'unknown')
    scan_type = scan_metadata.get('scan_type', 'unknown')
    src_to_iso = scan_metadata.get('source_origin', '')
    src_to_det = scan_metadata.get('origin_det', '')
    det_rows = scan_metadata.get('det_row_count', '')
    det_cols = scan_metadata.get('det_col_count', '')
    gantry_diam = scan_metadata.get('gantry_diameter', '')
    num_views = scan_metadata.get('num_views', '')
    view_range = scan_metadata.get('view_range', '')
    dosage = scan_metadata.get('dosage', '')
    recon_algo = scan_metadata.get('recon_algo', 'FBP')
    fov = scan_metadata.get('fov', '')
    anode_angle = scan_metadata.get('anode_angle', '')
    image_dims = scan_metadata.get('image_dims', [])
    num_spectra = scan_metadata.get('num_spectra', 1)
    det_spacing_x = scan_metadata.get('det_spacing_x', '')
    det_spacing_y = scan_metadata.get('det_spacing_y', '')
    dosage_list = scan_metadata.get('dosage_list', [])
    spectra_files = scan_metadata.get('spectra_files', [])
    img_scale = scan_metadata.get('img_scale', '')

    # ---- Build a TEMPLATE Dataset once with all series-invariant fields ----
    # All metadata that is constant across slices (patient, study, series,
    # scanner geometry, DECT tags, ...) goes here.  Per-slice we only patch
    # the few fields that actually vary: SOPInstanceUID, InstanceNumber,
    # ImagePositionPatient, SliceLocation, PixelData, Rows, Columns, and
    # file_meta.MediaStorageSOPInstanceUID.
    template = FileDataset('template.dcm', {}, preamble=b'\x00' * 128)

    template.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    template.StudyInstanceUID = study_uid
    template.SeriesInstanceUID = series_uid
    template.FrameOfReferenceUID = frame_uid

    template.PatientID = patient_id
    template.PatientName = patient_id

    template.StudyDate = date_str
    template.StudyTime = time_str
    template.StudyDescription = study_description
    template.AccessionNumber = ''
    template.ReferringPhysicianName = ''

    template.Modality = 'CT'
    template.SeriesDescription = series_description
    template.SeriesNumber = series_number

    template.Manufacturer = 'DEBISim2'
    template.InstitutionName = 'DEBISim2 Simulation'
    template.StationName = str(scanner_name)
    template.ManufacturerModelName = str(scanner_name)
    template.SoftwareVersions = 'DEBISim2 1.2.0'

    if kvp != '':
        template.KVP = str(kvp)
    if dosage != '':
        n_v = int(num_views) if num_views != '' else 1
        mAs = float(dosage) / (n_v if n_v > 0 else 1)
        template.XRayTubeCurrent = str(int(round(mAs)))
        template.Exposure = f'{float(dosage):.0f}'
        template.ExposureInuAs = int(float(dosage))
    if src_to_iso != '':
        template.DistanceSourceToPatient = f'{float(src_to_iso):.1f}'
    if src_to_det != '' and src_to_iso != '':
        template.DistanceSourceToDetector = \
            f'{float(src_to_iso) + float(src_to_det):.1f}'
    if gantry_diam != '':
        template.DataCollectionDiameter = f'{float(gantry_diam):.1f}'
    if fov != '':
        template.ReconstructionDiameter = f'{float(fov):.1f}'
    elif gantry_diam != '':
        template.ReconstructionDiameter = f'{float(gantry_diam):.1f}'

    if det_rows != '':
        template.NumberOfDetectorRows = int(det_rows)
    if det_cols != '':
        template.NumberOfDetectorColumns = int(det_cols)

    template.ConvolutionKernel = str(recon_algo).upper()
    template.FilterType = 'RAM-LAK'
    template.ContentDate = date_str

    _geom_to_dicom = {'PARALLEL': 'SEQUENCED', 'CONE': 'SPIRAL',
                      'FANBEAM': 'SEQUENCED'}
    template.AcquisitionType = _geom_to_dicom.get(
        str(geometry).upper(), 'SEQUENCED')
    template.ScanOptions = str(scan_type).upper()
    template.GantryDetectorTilt = '0.0'
    template.RotationDirection = 'CW'
    template.TableHeight = '0.0'

    if num_views != '':
        template.NumberOfProjections = int(num_views)
    if view_range != '':
        template.ScanArc = f'{float(view_range):.1f}'

    template.SpacingBetweenSlices = str(slice_thickness)

    if num_spectra > 1:
        template.add_new([0x0009, 0x0010], 'LO', 'DEBISim2_DECT')
        template.add_new([0x0009, 0x1001], 'IS', str(num_spectra))
        if spectra_files:
            template.add_new([0x0009, 0x1002], 'LO',
                             ', '.join(spectra_files))
        if dosage_list:
            template.add_new([0x0009, 0x1003], 'LO',
                             ', '.join(f'{d:.0f}' for d in dosage_list))
        if img_scale != '':
            template.add_new([0x0009, 0x1004], 'DS',
                             f'{float(img_scale):.6f}')

    template.WindowCenter = '40'
    template.WindowWidth = '400'

    # Coordinate transform tags (constant for the whole series)
    coord = DicomCoordinateMapper(pixel_spacing, slice_thickness)
    dicom_tags = coord.dicom_tags()
    template.ImageOrientationPatient = dicom_tags['ImageOrientationPatient']
    template.PixelSpacing = dicom_tags['PixelSpacing']
    template.SliceThickness = dicom_tags['SliceThickness']

    template.SamplesPerPixel = 1
    template.PhotometricInterpretation = 'MONOCHROME2'
    template.BitsAllocated = 16
    template.BitsStored = 16
    template.HighBit = 15
    template.PixelRepresentation = 1  # signed
    template.RescaleIntercept = '0'
    template.RescaleSlope = '1'
    template.RescaleType = 'HU'

    # File Meta template — SOPInstanceUID is patched per slice
    template.file_meta = Dataset()
    template.file_meta.MediaStorageSOPClassUID = template.SOPClassUID
    template.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    template.is_little_endian = True
    template.is_implicit_VR = False

    # ---- Per-slice: deepcopy template + patch only the varying fields -----
    # Deepcopy is ~5-10× faster than building from scratch.  Serialization
    # is offloaded to the archive's serializer pool (parallel across cores).
    for z in range(num_slices):
        ds = _copy.deepcopy(template)

        ds.SOPInstanceUID = sop_uids[z]
        ds.InstanceNumber = z + 1
        ds.ImagePositionPatient = coord.image_position_patient(z)
        ds.SliceLocation = str(z * slice_thickness)
        ds.file_meta.MediaStorageSOPInstanceUID = sop_uids[z]

        vol_slice = volume_3d[:, :, z]
        rows, cols = coord.pixel_rows_cols(vol_slice)
        ds.Rows = rows
        ds.Columns = cols
        ds.PixelData = coord.volume_slice_to_pixels(vol_slice).tobytes()

        slice_name = f'slice_{z:04d}.dcm'
        fname = os.path.join(output_dir, slice_name) if archive is None \
            else f'{output_dir}/{slice_name}'

        if archive is not None:
            # Hands the Dataset to the serializer pool — main thread
            # returns immediately, pydicom save_as runs on worker thread.
            archive.add_dicom(fname, ds)
        else:
            ds.save_as(fname)

    return dict(study_uid=study_uid, series_uid=series_uid,
                frame_uid=frame_uid, sop_uids=sop_uids)
# -----------------------------------------------------------------------------


# Category → RGB color mapping for RT-Struct ROI display
_ROI_COLORS = {
    'firearms':         [255, 0, 0],       # red
    'sharp_objects':    [255, 165, 0],     # orange
    'explosives':       [255, 255, 0],     # yellow
    'other':            [0, 255, 255],     # cyan
    'primitive_target': [255, 0, 255],     # magenta
    'liquid_container': [0, 128, 255],     # blue
    'filler':           [128, 128, 128],   # gray
    'primitive':        [128, 128, 128],   # gray
}


def create_rtstruct(output_path, roi_list, gt_label_volume,
                    ct_series_uids, pixel_spacing=(1.0, 1.0),
                    slice_thickness=1.0, patient_id='DEBISim',
                    recon_volume=None, archive=None):
    """
    Create a DICOM RT-Structure Set file with ROI contours for threats.

    :param output_path:      path for the output .dcm file (or archive arcname)
    :param roi_list:         list of dicts with keys: label, name, category, material
    :param gt_label_volume:  3D numpy int array (H x W x D) with object labels
    :param ct_series_uids:   dict from save_dicom_series (study_uid, series_uid, frame_uid, sop_uids)
    :param pixel_spacing:    (row_spacing, col_spacing) in mm
    :param slice_thickness:  slice thickness in mm
    :param patient_id:       DICOM PatientID
    :param archive:          optional MonolithicArchive — when set, output is
                             dispatched via archive.add_raw() instead of disk
    """
    import datetime
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.sequence import Sequence as DicomSequence
    from skimage.measure import find_contours

    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d')
    time_str = now.strftime('%H%M%S.%f')

    ds = FileDataset(output_path, {}, preamble=b'\x00' * 128)
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Storage
    ds.SOPInstanceUID = uid.generate_uid()
    ds.StudyInstanceUID = ct_series_uids['study_uid']
    ds.SeriesInstanceUID = uid.generate_uid()
    ds.SeriesNumber = '99'
    ds.Modality = 'RTSTRUCT'
    ds.Manufacturer = 'DEBISim2'
    ds.PatientID = patient_id
    ds.PatientName = patient_id
    ds.StructureSetLabel = 'ThreatROIs'
    ds.StructureSetDate = date_str
    ds.StructureSetTime = time_str

    # Referenced Frame of Reference → Referenced Study → Referenced Series
    ref_frame = Dataset()
    ref_frame.FrameOfReferenceUID = ct_series_uids['frame_uid']

    ref_study = Dataset()
    ref_study.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.1'
    ref_study.ReferencedSOPInstanceUID = ct_series_uids['study_uid']

    ref_series = Dataset()
    ref_series.SeriesInstanceUID = ct_series_uids['series_uid']
    contour_image_seq = []
    for sop_uid in ct_series_uids['sop_uids']:
        ci = Dataset()
        ci.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        ci.ReferencedSOPInstanceUID = sop_uid
        contour_image_seq.append(ci)
    ref_series.ContourImageSequence = DicomSequence(contour_image_seq)

    ref_study.RTReferencedSeriesSequence = DicomSequence([ref_series])
    ref_frame.RTReferencedStudySequence = DicomSequence([ref_study])
    ds.ReferencedFrameOfReferenceSequence = DicomSequence([ref_frame])

    # Build ROI sequences
    structure_set_roi_seq = []
    roi_contour_seq = []
    rt_roi_obs_seq = []

    num_slices = gt_label_volume.shape[2]

    # Only emit contours on slices where the reconstruction has content.
    # This prevents contours from extending beyond the FBP valid range.
    if recon_volume is not None:
        _content_thresh = -990  # HU above air
        _valid_slices = set()
        for z in range(num_slices):
            if recon_volume[:, :, z].max() > _content_thresh:
                _valid_slices.add(z)
    else:
        _valid_slices = set(range(num_slices))

    for roi_num, roi in enumerate(roi_list, start=1):
        label = roi['label']
        category = roi.get('category', 'unknown')
        color = _ROI_COLORS.get(category, [200, 200, 200])

        # StructureSetROISequence entry
        ss_roi = Dataset()
        ss_roi.ROINumber = roi_num
        ss_roi.ReferencedFrameOfReferenceUID = ct_series_uids['frame_uid']
        ss_roi.ROIName = roi['name']
        ss_roi.ROIGenerationAlgorithm = 'AUTOMATIC'
        structure_set_roi_seq.append(ss_roi)

        # ROIContourSequence entry — extract contours per slice
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = color
        roi_contour.ReferencedROINumber = roi_num
        contour_seq = []

        for z in range(num_slices):
            if z not in _valid_slices:
                continue
            mask_slice = (gt_label_volume[:, :, z] == label).astype(np.uint8)
            if mask_slice.sum() == 0:
                continue
            contours = find_contours(mask_slice, 0.5)
            coord = DicomCoordinateMapper(pixel_spacing, slice_thickness)
            for contour in contours:
                # Convert GT voxel indices to patient coords via the
                # shared DicomCoordinateMapper (same transform used
                # by save_dicom_series for pixel data).
                contour_data = []
                for row, col in contour:
                    x, y, z_pos = coord.voxel_to_patient(row, col, z)
                    contour_data.extend([x, y, z_pos])

                c_item = Dataset()
                c_item.ContourGeometricType = 'CLOSED_PLANAR'
                c_item.NumberOfContourPoints = len(contour)
                c_item.ContourData = contour_data

                # Reference the CT slice
                c_image_ref = Dataset()
                c_image_ref.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
                c_image_ref.ReferencedSOPInstanceUID = ct_series_uids['sop_uids'][z]
                c_item.ContourImageSequence = DicomSequence([c_image_ref])

                contour_seq.append(c_item)

        roi_contour.ContourSequence = DicomSequence(contour_seq)
        roi_contour_seq.append(roi_contour)

        # RTROIObservationsSequence entry
        obs = Dataset()
        obs.ObservationNumber = roi_num
        obs.ReferencedROINumber = roi_num
        obs.RTROIInterpretedType = 'GTV' if roi.get('threat', False) else 'ORGAN'
        obs.ROIInterpreter = ''
        # Store material properties as ROI description
        mat_info = roi.get('material', '')
        mu_val = roi.get('mu', '')
        z_val = roi.get('z_eff', '')
        dens = roi.get('density', '')
        obs.ROIObservationDescription = (
            f'material={mat_info}; mu={mu_val}; z_eff={z_val}; '
            f'density={dens}; threat={roi.get("threat", False)}')
        rt_roi_obs_seq.append(obs)

    ds.StructureSetROISequence = DicomSequence(structure_set_roi_seq)
    ds.ROIContourSequence = DicomSequence(roi_contour_seq)
    ds.RTROIObservationsSequence = DicomSequence(rt_roi_obs_seq)

    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    if archive is not None:
        buf = io.BytesIO()
        ds.save_as(buf)
        archive.add_raw(output_path, buf.getvalue())
    else:
        ds.save_as(output_path)
# -----------------------------------------------------------------------------


def save_dicom_output(image_dir, recon_images, gt_label_volume,
                      sf_obj_list, pixel_spacing=(1.0, 1.0),
                      slice_thickness=1.0, scan_metadata=None,
                      mu_handler=None, archive=None):
    """
    High-level function: save DICOM CT series + RT-Struct with ROIs for all
    objects (threats and non-threats alike).

    :param image_dir:        base output directory (e.g., simulation_XXX/images/)
    :param recon_images:     dict mapping name to 3D numpy array (e.g. {'recon_1': arr})
    :param gt_label_volume:  3D int array with per-object labels
    :param sf_obj_list:      list of object metadata dicts (with category/threat keys)
    :param pixel_spacing:    (row, col) spacing in mm
    :param slice_thickness:  slice thickness in mm
    :param scan_metadata:    dict of scanner/source parameters for DICOM tags
    :param mu_handler:       MuDatabaseHandler instance for material property lookups
    :param archive:          optional MonolithicArchive — when set, all DICOM
                             output is serialized to bytes and dispatched via
                             archive.add_raw() instead of writing to disk
    """
    # When archiving, use forward-slash joins for arcnames (not os.path.join
    # which produces backslashes on Windows).
    if archive is not None:
        _join = lambda *parts: '/'.join(parts)
    else:
        _join = os.path.join

    dicom_dir = _join(image_dir, 'dicom')
    if archive is None:
        os.makedirs(dicom_dir, exist_ok=True)

    # Save each recon image as a DICOM CT series
    ct_uids = None
    for idx, (name, volume) in enumerate(recon_images.items(), start=1):
        series_dir = _join(dicom_dir, f'series_{name}')
        # Per-series metadata: override kVp from series name if it has kV
        series_meta = dict(scan_metadata) if scan_metadata else {}
        import re
        kv_match = re.search(r'(\d+)\s*[kK][vV]', name)
        if kv_match:
            series_meta['kVp'] = int(kv_match.group(1))
        uids = save_dicom_series(
            series_dir, volume,
            series_description=name, series_number=idx,
            pixel_spacing=pixel_spacing, slice_thickness=slice_thickness,
            scan_metadata=series_meta, archive=archive
        )
        if ct_uids is None:
            ct_uids = uids  # use first series as reference for RT-Struct

    if ct_uids is None:
        return

    # Build ROI list from ALL objects (threats + fillers + liquid containers
    # + primitives) — each annotated with material, mu, z_eff
    roi_list = []
    for obj in sf_obj_list:
        cat = obj.get('category', 'primitive')
        mat = obj.get('material', 'unknown')
        label = obj.get('label', 0)
        is_threat = obj.get('threat', False)

        # Look up material properties from the mu database
        mu_val = ''
        z_eff = ''
        density = ''
        if mu_handler is not None and mat != 'unknown':
            try:
                mat_dict = mu_handler.material(mat)
                # mu is an array over keV — store the mean LAC (density * mean mu)
                mu_arr = mat_dict.get('mu', None)
                density = mat_dict.get('density', '')
                z_eff = mat_dict.get('z', '')
                if mu_arr is not None and density != '':
                    mu_val = float(np.mean(mu_arr) * density)
                if z_eff != '':
                    z_eff = float(z_eff)
                if density != '':
                    density = float(density)
            except Exception:
                pass

        roi_list.append(dict(
            label=label,
            name=f'{cat}_{label}_{mat}',
            category=cat,
            material=mat,
            threat=is_threat,
            mu=mu_val,
            z_eff=z_eff,
            density=density,
        ))

    if len(roi_list) > 0:
        rtstruct_path = _join(dicom_dir, 'rtstruct.dcm')
        # Use the first recon volume to clip contours to valid FBP slices
        ref_recon = next(iter(recon_images.values()), None)
        create_rtstruct(
            rtstruct_path, roi_list, gt_label_volume,
            ct_uids, pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            recon_volume=ref_recon, archive=archive,
        )

        # Save a human-readable ROI manifest (CSV) alongside the DICOM files
        import csv
        manifest_path = _join(dicom_dir, 'roi_manifest.csv')
        fieldnames = ['label', 'name', 'category', 'material', 'threat',
                      'mu', 'z_eff', 'density']
        if archive is not None:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            for roi in roi_list:
                writer.writerow(roi)
            archive.add_raw(manifest_path, buf.getvalue().encode('utf-8'))
        else:
            with open(manifest_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for roi in roi_list:
                    writer.writerow(roi)
# -----------------------------------------------------------------------------


def create_pil_collage(images, fpath, layout=None, vlims=None):
    """
    ---------------------------------------------------------------------------

    :param images:
    :param fpath:
    :param layout:
    :param vlims:
    :return:
    ---------------------------------------------------------------------------
    """

    if layout is None: layout = (len(images), 1)

    if vlims is not None:
        images = [np.clip(x, vlims[0], vlims[1])/vlims[1]*255 for x in images]
    else:
        images = [x/x.max()*255 for x in images]

    assert layout[0]*layout[1] == len(images)

    rows, cols = layout

    collage = np.vstack([np.hstack(images[x*cols:(x+1)*cols])
                         for x in range(rows)])
    im = Image.fromarray(collage)
    im = im.convert('L')
    im.save(fpath)
# -------------------------------------------------------------------------



