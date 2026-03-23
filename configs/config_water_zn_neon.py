# Three-material test: water, zinc, neoprene.
#
# Spans low-Z (water, Z=7.4), mid-Z (neoprene, Z=12.4), high-Z (Zn, Z=30).
# Large cylinders (30 voxels = 60mm) for clean interior sampling.
# No streak artifacts from neighboring dense objects.

from lib.__init__ import *
from lib.forward_model.scanner_template import create_parallel_scanner

# ---- Dataset ----------------------------------------------------------------
bags_to_create = range(1, 2)
sim_dir        = 'results/water_zn_neoprene/'

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

# ---- 3 materials in a row ---------------------------------------------------
phantom_materials = ['water', 'Zn', 'neoprene']

bag_creator_args = dict(
    phantom_mode=True,
    phantom_materials=phantom_materials,
    phantom_grid=(1, 3),
    phantom_block_size=30,
    phantom_gap=20,
    material_list=phantom_materials,
    liquid_list=[],
    material_pdf=[1.0 / len(phantom_materials)] * len(phantom_materials),
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
params['save_sino']         = True
params['basis_fn']          = None
params['decomposer_args']   = cdm_args
params['recon_args']        = None
params['images_to_save']    = ['gt', 'lac_1', 'lac_2', 'compton', 'pe', 'zeff']
params['decomposer']        = decomp_method
params['slicewise']         = False
params['dicom_output']      = True
params['compress_dicom']    = False
