# Water cylinder only — minimal config for HU calibration validation.
#
# Single thin-walled water cylinder centered in the FOV.
# No dense materials, no metals, no beam hardening complications.
# Water should reconstruct to ~0 HU if the pipeline is calibrated.
#
# Use small block_size (5 voxels = 10mm diameter at 2mm/voxel)
# to minimize beam hardening through the cylinder.

from lib.__init__ import *
from lib.forward_model.scanner_template import create_parallel_scanner

# ---- Dataset ----------------------------------------------------------------
bags_to_create = range(1, 2)
sim_dir        = 'results/water_cylinder/'

# ---- Scanner ----------------------------------------------------------------
scanner_mdl = create_parallel_scanner(
    gantry_diameter_mm=512,
    pixel_size_mm=2.0,
    n_slices=64,
    n_views=720,
)

# ---- Dual-energy source: airport DECT --------------------------------------
xray_source_specs = dict(
    num_spectra=2,
    kVp=160,
    spectra=[os.path.join(SPECTRA_DIR, 'airport_spectrum_160kV.txt'),
             os.path.join(SPECTRA_DIR, 'airport_spectrum_80kV.txt')],
    dosage=[5e6, 4e6],
)

# ---- Single water cylinder --------------------------------------------------
phantom_materials = ['water']

bag_creator_args = dict(
    phantom_mode=True,
    phantom_materials=phantom_materials,
    phantom_grid=(1, 1),
    phantom_block_size=30,      # 30 voxels = 60mm diameter — enough interior voxels
    phantom_gap=0,
    material_list=phantom_materials,
    liquid_list=[],
    material_pdf=[1.0],
    liquid_pdf=[],
    prevent_overlap=False,
)

# ---- Decomposition ----------------------------------------------------------
decomp_method = 'cdm'
cdm_args = dict(
    cdm_solver='gpu',
    cdm_type='cpd',
    projector='gpu',
    init_val=(0.1, 0.1),
)

# ---- Pipeline params --------------------------------------------------------
params = dict()
params['num_bags']          = bags_to_create
params['sim_dir']           = sim_dir
params['scanner']           = scanner_mdl
params['xray_src_mdl']      = xray_source_specs
params['bag_creator_args']  = bag_creator_args
params['save_sino']         = False
params['basis_fn']          = None
params['decomposer_args']   = cdm_args
params['recon_args']        = None
params['images_to_save']    = ['gt', 'lac_1']
params['decomposer']        = decomp_method
params['slicewise']         = False
params['dicom_output']      = True
params['compress_dicom']    = False
