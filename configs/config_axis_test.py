# Axis orientation test: 3 materials at asymmetric positions.
#
# Layout (top-down view):
#   Water at grid (0,0) = top-left
#   Zn at grid (2,2) = bottom-right
#   Neoprene at grid (2,0) = bottom-left
#
# If X is flipped: water and neoprene swap columns
# If Y is flipped: water moves to bottom-left, neoprene to top-left
# If X-Y swapped: diagonal mirror
#
# The test: sample HU at each GT label centroid. Water should be
# lowest HU, Zn highest. If they're swapped, an axis is wrong.

from lib.__init__ import *
from lib.forward_model.scanner_template import create_parallel_scanner

bags_to_create = range(1, 2)
sim_dir        = 'results/axis_test/'

scanner_mdl = create_parallel_scanner(
    gantry_diameter_mm=512,
    pixel_size_mm=2.0,
    n_slices=64,
    n_views=720,
)

xray_source_specs = dict(
    num_spectra=2,
    kVp=160,
    spectra=[os.path.join(SPECTRA_DIR, 'airport_spectrum_160kV.txt'),
             os.path.join(SPECTRA_DIR, 'airport_spectrum_80kV.txt')],
    dosage=[5e6, 4e6],
)

# 3x3 grid, only 3 corners populated — maximally asymmetric
phantom_materials = [
    'water',    None,       None,
    None,       None,       None,
    'neoprene', None,       'Zn',
]
# Filter out Nones — the phantom creator needs a flat list
# Use air for empty slots so positions are deterministic
phantom_materials = [
    'air',      'air',      'air',
    'air',      'air',      'air',
    'neoprene', 'air',      'Zn',
]
# Put water at (0,0), neoprene at (2,0), Zn at (2,2)
phantom_materials[0] = 'water'

bag_creator_args = dict(
    phantom_mode=True,
    phantom_materials=phantom_materials,
    phantom_grid=(3, 3),
    phantom_block_size=20,
    phantom_gap=30,
    material_list=['water', 'air', 'neoprene', 'Zn'],
    liquid_list=[],
    material_pdf=[0.25, 0.25, 0.25, 0.25],
    liquid_pdf=[],
    prevent_overlap=False,
)

decomp_method = 'cdm'
cdm_args = dict(
    cdm_solver='gpu',
    cdm_type='cpd',
    projector='gpu',
    init_val=(0.1, 0.1),
)

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
