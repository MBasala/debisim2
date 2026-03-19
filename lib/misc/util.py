import matplotlib as mpl

# mpl.use('TkAgg')
from pylab import *
from numpy import *

import os, pickle, pydicom
import scipy.sparse as sp
import scipy.misc as misc
from concurrent.futures import ThreadPoolExecutor
import threading

from matplotlib.patches import Ellipse
from astropy.io import fits as pyfits
from pydicom import uid
from skimage.measure import regionprops
from skimage.transform import rescale

from lib.__init__ import *
from lib.misc.multi_processor import *
from PIL import Image
import imageio


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

    # Create file handler
    f_handler = logging.FileHandler(logfile, mode='a')
    s_handler = logging.StreamHandler(sys.stdout)

    # Create formatter
    formatter = logging.Formatter(
                        '[%(asctime)s] [%(name)s] %(levelname)s: %(message)s')
    f_handler.setFormatter(formatter)
    s_handler.setFormatter(formatter)

    logger.addHandler(f_handler)
    logger.addHandler(s_handler)
    logger.propagate = False

    return logger
# -----------------------------------------------------------------------------


def quick_imshow(nrows, ncols=1,
                 images=None,
                 titles=None,
                 colorbar=True,
                 vmax=None,
                 vmin=None,
                 figsize=None,
                 figtitle=None,
                 visibleaxis=False,
                 colormap='jet',
                 saveas=''):
    """-------------------------------------------------------------------------
    Convenience function that make subplots of imshow

    :param  nrows - number of rows
    :param  ncols - number of cols
    :param  images - list of images
    :param  titles - list of titles
    :param  vmax - tuple of vmax for the colormap. If scalar, the same value is
                   used for all subplots. If one of the entries is None, no
                   colormap for that subplot will be drawn.
    :param  vmin - tuple of vmin

    :return: f - the figure handle
             axes - axes or array of axes objects
             caxes - tuple of axes image
    -------------------------------------------------------------------------"""

    if isinstance(nrows, ndarray):
        images = nrows
        nrows = 1
        ncols = 1

    if figsize == None:
        s = 3.5
        if figtitle:
            figsize = (s * ncols, s * nrows + 0.5)
        else:
            figsize = (s * ncols, s * nrows)

    if nrows == ncols == 1:
        f, ax = subplots(figsize=figsize)
        cax = ax.imshow(images, cmap=colormap, vmax=vmax, vmin=vmin)
        if colorbar:
            f.colorbar(cax, ax=ax)
        if titles != None:
            ax.set_title(titles)
        if figtitle != None:
            f.suptitle(figtitle)
        cax.axes.get_xaxis().set_visible(visibleaxis)
        cax.axes.get_yaxis().set_visible(visibleaxis)
        return f, ax, cax

    f, axes = subplots(nrows, ncols, figsize=figsize)
    caxes = []
    i = 0
    for ax, img in zip(axes.flat, images):
        if isinstance(vmax, tuple) and isinstance(vmin, tuple):
            if vmax[i] is not None and vmin[i] is not None:
                cax = ax.imshow(img, cmap=colormap, vmax=vmax[i], vmin=vmin[i])
            else:
                cax = ax.imshow(img, cmap=colormap)
        elif isinstance(vmax, tuple) and vmin is None:
            if vmax[i] is not None:
                cax = ax.imshow(img, cmap=colormap, vmax=vmax[i], vmin=0)
            else:
                cax = ax.imshow(img, cmap=colormap)
        elif vmax is None and vmin is None:
            cax = ax.imshow(img, cmap=colormap)
        else:
            cax = ax.imshow(img, cmap=colormap, vmax=vmax, vmin=vmin)
        if titles != None:
            ax.set_title(titles[i])
        if colorbar:
            f.colorbar(cax, ax=ax)
        caxes.append(cax)
        cax.axes.get_xaxis().set_visible(visibleaxis)
        cax.axes.get_yaxis().set_visible(visibleaxis)
        i = i + 1
    if figtitle != None:
        f.suptitle(figtitle)
    if saveas != '':
        f.savefig(saveas)
    return f, axes, tuple(caxes)
# ------------------------------------------------------------------------------


def update_subplots(images,
                    caxes,
                    f=None,
                    axes=None,
                    indices=(),
                    vmax=None,
                    vmin=None):
    """
    ----------------------------------------------------------------------------
    Update subplots in a figure

    :param images  - new images to plot
    :param caxes   - caxes returned at figure creation
    :param indices - specific indices of subplots to be updated

    :return
    ----------------------------------------------------------------------------
    """

    for i in range(len(images)):
        if len(indices) > 0:
            ind = indices[i]
        else:
            ind = i
        img = images[i]
        caxes[ind].set_data(img)
        cbar = caxes[ind].colorbar
        if isinstance(vmax, tuple) and isinstance(vmin, tuple):
            if vmax[i] is not None and vmin[i] is not None:
                cbar.set_clim([vmin[i], vmax[i]])
            else:
                cbar.set_clim([img.min(), img.max()])
        elif isinstance(vmax, tuple) and vmin is None:
            if vmax[i] is not None:
                cbar.set_clim([0, vmax[i]])
            else:
                cbar.set_clim([img.min(), img.max()])
        elif vmax is None and vmin is None:
            cbar.set_clim([img.min(), img.max()])
        else:
            cbar.set_clim([vmin, vmax])
        cbar.update_normal(caxes[ind])

    pause(0.01)
    tight_layout()
# ------------------------------------------------------------------------------


def slide_show(image, dt=0.01, vmax=None, vmin=None):
    """
    ---------------------------------------------------------------------------
    Slide show for visualizing an image volume. Image is (w, h, d)

    :param image: (w, h, d), slides are 2D images along the depth axis
    :param dt:      transition time
    :param vmax:    maximum cliiping value
    :param vmin:    minimum clipping value
    :return:
    ---------------------------------------------------------------------------
    """

    if image.dtype == bool:
        image *= 1.0
    if vmax is None:
        vmax = image.max()
    if vmin is None:
        vmin = image.min()
    plt.ion()
    plt.figure()
    for i in arange(image.shape[2]):
        plt.cla()
        cax = plt.imshow(image[:, :, i], cmap='jet', vmin=vmin, vmax=vmax)
        plt.title(str('Slice: %i' % i))
        if i == 0:
            cf = plt.gcf()
            ca = plt.gca()
            cf.colorbar(cax, ax=ca)
        plt.pause(dt)
        plt.draw()
# -----------------------------------------------------------------------------


def scatter_ellipse(X, labels, mu, R, figsize=(5, 5), s=0.01, alpha=0.1):
    """
    ---------------------------------------------------------------------------
    2D scatter plot with ellipse drawn based on mean and covariance.

    :param X: samples, (N, 2)
    :param labels: integer labels, (N,)
    :param mu: centroids, (k, 2)
    :param R: covariances, (k, 2, 2)
    :return:
    ---------------------------------------------------------------------------
    """
    k = len(unique(labels))

    f, ax = subplots(figsize=figsize)
    ax.scatter(X[:, 0], X[:, 1],
               s=s, c=labels, alpha=alpha, cmap='jet')

    for m in range(k):
        vals, vecs = eigh(R[m])
        x, y = vecs[:, 0]
        w, h = 2 * sqrt(vals)
        theta = degrees(arctan2(y, x))
        ax.add_artist(
            Ellipse(xy=mu[m],
                    width=w,
                    height=h,
                    angle=theta,
                    fill=False,
                    edgecolor='r'))

    return f, ax
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


def save_fits_data_async(file_path, out_image, compress=False):
    """
    Submit a FITS save to the background I/O thread pool.

    The image array is copied before submission so the caller can safely
    reuse or free the memory immediately.

    :param file_path:   path to the fits file
    :param out_image:   numpy array to save
    :param compress:    whether to use FITS compression
    """
    data_copy = out_image.copy()
    fut = _get_io_pool().submit(save_fits_data, file_path, data_copy, compress)
    with _io_lock:
        # Prune completed futures to avoid unbounded growth
        _io_futures[:] = [f for f in _io_futures if not f.done()]
        _io_futures.append(fut)


def flush_async_io():
    """
    Block until all pending async FITS writes have completed.
    Call this before the pipeline exits or before reading back saved files.
    Raises the first exception encountered, if any.
    """
    with _io_lock:
        pending = list(_io_futures)
        _io_futures.clear()

    for fut in pending:
        fut.result()  # raises if the write failed
# -----------------------------------------------------------------------------


def save_dicom_series(output_dir, volume_3d, patient_id='DEBISim',
                      study_description='Simulated CT',
                      series_description='Recon', series_number=1,
                      pixel_spacing=(1.0, 1.0), slice_thickness=1.0,
                      scan_metadata=None):
    """
    Save a 3D numpy array as a DICOM CT image series (one .dcm per slice).

    :param output_dir:  directory to write .dcm files into
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
    :returns: dict with keys study_uid, series_uid, frame_uid, sop_uids
    """
    import datetime
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.sequence import Sequence as DicomSequence

    if scan_metadata is None:
        scan_metadata = {}

    os.makedirs(output_dir, exist_ok=True)

    study_uid = uid.generate_uid()
    series_uid = uid.generate_uid()
    frame_uid = uid.generate_uid()
    sop_uids = []

    num_slices = volume_3d.shape[2]
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
    dosage = scan_metadata.get('dosage', '')
    recon_algo = scan_metadata.get('recon_algo', 'FBP')
    fov = scan_metadata.get('fov', '')
    anode_angle = scan_metadata.get('anode_angle', '')
    image_dims = scan_metadata.get('image_dims', [])

    for z in range(num_slices):
        sop_uid = uid.generate_uid()
        sop_uids.append(sop_uid)

        fname = os.path.join(output_dir, f'slice_{z:04d}.dcm')
        ds = FileDataset(fname, {}, preamble=b'\x00' * 128)

        # --- SOP / Study / Series UIDs ---
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        ds.SOPInstanceUID = sop_uid
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid

        # --- Patient Module ---
        ds.PatientID = patient_id
        ds.PatientName = patient_id

        # --- General Study Module ---
        ds.StudyDate = date_str
        ds.StudyTime = time_str
        ds.StudyDescription = study_description
        ds.AccessionNumber = ''
        ds.ReferringPhysicianName = ''

        # --- General Series Module ---
        ds.Modality = 'CT'
        ds.SeriesDescription = series_description
        ds.SeriesNumber = series_number
        ds.InstanceNumber = z + 1

        # --- General Equipment Module ---
        ds.Manufacturer = 'DEBISim2'
        ds.InstitutionName = 'DEBISim2 Simulation'
        ds.StationName = str(scanner_name)
        ds.ManufacturerModelName = str(scanner_name)
        ds.SoftwareVersions = 'DEBISim2 1.2.0'

        # --- CT Image Module (scan acquisition parameters) ---
        if kvp != '':
            ds.KVP = str(kvp)
        if dosage != '':
            ds.XRayTubeCurrent = str(int(float(dosage)))
            ds.Exposure = str(int(float(dosage)))
        if src_to_iso != '':
            ds.DistanceSourceToPatient = str(float(src_to_iso))
        if src_to_det != '' and src_to_iso != '':
            ds.DistanceSourceToDetector = str(float(src_to_iso) + float(src_to_det))
        if gantry_diam != '':
            ds.ReconstructionDiameter = str(float(gantry_diam))
            ds.DataCollectionDiameter = str(float(gantry_diam))
        if fov != '':
            ds.ReconstructionDiameter = str(float(fov))
        if det_rows != '':
            ds.NumberOfDetectorRows = str(det_rows)
        if det_cols != '':
            ds.NumberOfDetectorColumns = str(det_cols)
        # Note: anode_angle is not stored — DICOM FocalSpotSize is a physical
        # size (mm), not an angle.  NumberOfFrames is omitted because each
        # file is a single-frame CT slice.
        ds.ConvolutionKernel = str(recon_algo)
        ds.FilterType = str(recon_algo)
        ds.ContentDate = date_str

        # Geometry info in private/protocol tags
        ds.AcquisitionType = str(geometry).upper()
        ds.ScanOptions = str(scan_type).upper()

        # Window center/width for typical CT display
        ds.WindowCenter = '40'
        ds.WindowWidth = '400'

        # --- Image Pixel Module ---
        ds.ImagePositionPatient = [0.0, 0.0, float(z * slice_thickness)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = list(pixel_spacing)
        ds.SliceThickness = str(slice_thickness)
        ds.SliceLocation = str(z * slice_thickness)

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.Rows = volume_3d.shape[0]
        ds.Columns = volume_3d.shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # signed
        ds.RescaleIntercept = '0'
        ds.RescaleSlope = '1'
        ds.RescaleType = 'HU'

        # Clip to int16 range before casting to prevent wrap-around
        slice_data = volume_3d[:, :, z]
        pixel_data = np.clip(slice_data, -32768, 32767).astype(np.int16)
        ds.PixelData = pixel_data.tobytes()

        # --- File Meta ---
        ds.file_meta = Dataset()
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
        ds.is_little_endian = True
        ds.is_implicit_VR = False

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
                    slice_thickness=1.0, patient_id='DEBISim'):
    """
    Create a DICOM RT-Structure Set file with ROI contours for threats.

    :param output_path:      path for the output .dcm file
    :param roi_list:         list of dicts with keys: label, name, category, material
    :param gt_label_volume:  3D numpy int array (H x W x D) with object labels
    :param ct_series_uids:   dict from save_dicom_series (study_uid, series_uid, frame_uid, sop_uids)
    :param pixel_spacing:    (row_spacing, col_spacing) in mm
    :param slice_thickness:  slice thickness in mm
    :param patient_id:       DICOM PatientID
    """
    import datetime
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.sequence import Sequence as DicomSequence
    from skimage.measure import find_contours

    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d')
    time_str = now.strftime('%H%M%S.%f')

    ds = FileDataset(output_path, {}, preamble=b'\x00' * 128)
    ds.SOPClassUID = '1.2.840.10008.5.1.1.481.3'  # RT Structure Set Storage
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
            mask_slice = (gt_label_volume[:, :, z] == label).astype(np.uint8)
            if mask_slice.sum() == 0:
                continue
            contours = find_contours(mask_slice, 0.5)
            for contour in contours:
                # Convert pixel coords (row, col) to patient coords (x, y, z)
                contour_data = []
                for row, col in contour:
                    x = col * pixel_spacing[1]
                    y = row * pixel_spacing[0]
                    z_pos = z * slice_thickness
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

    ds.save_as(output_path)
# -----------------------------------------------------------------------------


def save_dicom_output(image_dir, recon_images, gt_label_volume,
                      sf_obj_list, pixel_spacing=(1.0, 1.0),
                      slice_thickness=1.0, scan_metadata=None,
                      mu_handler=None):
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
    """
    dicom_dir = os.path.join(image_dir, 'dicom')
    os.makedirs(dicom_dir, exist_ok=True)

    # Save each recon image as a DICOM CT series
    ct_uids = None
    for idx, (name, volume) in enumerate(recon_images.items(), start=1):
        series_dir = os.path.join(dicom_dir, f'series_{name}')
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
            scan_metadata=series_meta
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
        rtstruct_path = os.path.join(dicom_dir, 'rtstruct.dcm')
        create_rtstruct(
            rtstruct_path, roi_list, gt_label_volume,
            ct_uids, pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness
        )

        # Save a human-readable ROI manifest (CSV) alongside the DICOM files
        import csv
        manifest_path = os.path.join(dicom_dir, 'roi_manifest.csv')
        fieldnames = ['label', 'name', 'category', 'material', 'threat',
                      'mu', 'z_eff', 'density']
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


def create_gif(fname, input_vol, stride=1, scale=None):
    """

    :param fname:
    :param input_vol:
    :param stride:
    :return:
    """

    input_vol = (input_vol-input_vol.min())/(input_vol.max()-input_vol.min())
    input_vol = (input_vol*255).astype(uint8)

    if scale is None:
        imageio.mimsave(fname,
                        [input_vol[:,:, z]
                         for z in range(0,input_vol.shape[2], stride)],
                        fps=5)
    else:
        imageio.mimsave(fname,
                        [rescale(input_vol[:,:, z], scale=scale, preserve_range=True)
                         for z in range(0,input_vol.shape[2], stride)],
                        fps=5)

