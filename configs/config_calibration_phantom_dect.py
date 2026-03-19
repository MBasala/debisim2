# -----------------------------------------------------------------------------
"""
Calibration phantom configuration for dual-energy CT.

Produces a 5x5 grid of homogeneous material blocks in a single slice plane.
Each block has a unique DICOM ROI with material name, mu, z_eff, and density.
Used for algorithm validation against known ground-truth material properties.

Scanner: default parallel-beam, dual-energy (130 kV / 95 kV)
"""
# -----------------------------------------------------------------------------

from lib.__init__ import *
from lib.forward_model.scanner_template import create_parallel_scanner

# -----------------------------------------------------------------------------
# Step 1: Dataset parameters

bags_to_create = range(1, 2)                          # single phantom
sim_dir        = 'results/calibration_phantom_dect/'
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 2: Scanner model
#
# gantry_diameter_mm = physical FOV diameter (mm)
# pixel_size_mm      = reconstructed voxel size (mm)

scanner_mdl = create_parallel_scanner(
    gantry_diameter_mm=1024,
    pixel_size_mm=2.0,
    n_slices=350,
)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 3: Dual-energy X-ray source (130 kV / 95 kV)

xray_source_specs = dict(
    num_spectra=2,
    kVp=130,
    spectra=[os.path.join(SPECTRA_DIR, 'example_spectrum_130kV.txt'),
             os.path.join(SPECTRA_DIR, 'example_spectrum_95kV.txt')],
    dosage=[2e5, 1.85e5]
)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 4: Calibration phantom — 5x5 grid of material blocks
#
# Materials are ordered row-major (row 0 left→right, then row 1, etc.)
# and cover a wide range of effective atomic numbers:
#
#   Row 0  (metals, high Z):  Al   Ti   Fe   Cu   Zn
#   Row 1  (compounds):       water  acetal  acrylic  bakelite  bone
#   Row 2  (plastics/rubber): neoprene  nylon6  pvc  polyethylene  polystyrene
#   Row 3  (ceramics/misc):   pyrex  salt  teflon  polyester  Si
#   Row 4  (organics/low-Z):  ethanol  H2O2  saline035  C  air

phantom_materials = [
    'Al',    'Ti',    'Fe',         'Cu',          'Zn',
    'water', 'acetal', 'acrylic',   'bakelite',    'bone',
    'neoprene', 'nylon6', 'pvc',   'polyethylene', 'polystyrene',
    'pyrex', 'salt',  'teflon',    'polyester',    'Si',
    'ethanol', 'H2O2', 'saline035', 'C',           'air',
]

bag_creator_args = dict(
    phantom_mode=True,
    phantom_materials=phantom_materials,
    phantom_grid=(5, 5),         # rows x cols
    phantom_block_size=30,       # voxels (= 30 mm at 1 mm/voxel)
    phantom_gap=20,              # voxels between blocks (= 20 mm)
    # Required by pipeline even though random generation is skipped:
    material_list=phantom_materials,
    liquid_list=[],
    material_pdf=[1.0 / len(phantom_materials)] * len(phantom_materials),
    liquid_pdf=[],
    prevent_overlap=False,
)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 5: Dual-energy decomposition

decomp_method = 'cdm'
cdm_args = dict(cdm_solver='gpu',
                cdm_type='cpd',
                projector='gpu',
                init_val=(0, 0))
# -----------------------------------------------------------------------------

# Assemble params dictionary
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
params['images_to_save']    = ['gt', 'lac_1', 'lac_2',
                               'compton', 'pe', 'zeff']
params['decomposer']        = decomp_method
params['slicewise']         = False
params['dicom_output']      = True
# -----------------------------------------------------------------------------
