# Five-cylinder explosive threat phantom — sinogram + ground truth + DICOM + RT-Struct.
#
# Geometry: five 3 cm-diameter cylinders arranged in a single row inside a
# 512 mm parallel-beam FOV.  Each cylinder contains a distinct threat or
# reference liquid:
#
#   C1  water        — inert reference (Z≈7.43, ρ=1.00 g/cm³)
#   C2  rdx          — solid RDX explosive (Z≈7.21, ρ=1.82 g/cm³)
#   C3  h2o2_30      — 30 wt% hydrogen peroxide (Z≈7.50, ρ=1.11 g/cm³)
#   C4  tatp         — TATP granules (Z≈6.70, ρ=0.95 g/cm³)
#   C5  rdx_water    — RDX/water slurry 50/50 wt% (Z≈7.32, ρ=1.29 g/cm³)
#
# Scanner: 1 mm pixels, 512 mm gantry, 720 views, 64 slices.
# Cylinders are 30 voxels (30 mm) diameter; centred in the volume.
#
# Outputs:
#   save_sino=True            → sinograms (FITS)
#   images_to_save            → gt, lac_1, lac_2, compton, pe, zeff
#   dicom_output=True         → DICOM CT series + RT-Struct per cylinder
#   compress_dicom=True       → single .tar.gz archive

from lib.__init__ import *
from lib.forward_model.scanner_template import create_parallel_scanner

# ---- Dataset ----------------------------------------------------------------
bags_to_create = range(1, 2)
sim_dir        = 'results/explosive_cylinders/'

# ---- Scanner: 1 mm/pixel, 512 mm FOV, 64 slices ----------------------------
# 3 cm cylinders need at least 30 voxels diameter.
# 5 cylinders at 30 voxels + 15 voxel gaps → 5*30 + 4*15 = 210 mm — well within 512 mm.
scanner_mdl = create_parallel_scanner(
    gantry_diameter_mm=512,
    pixel_size_mm=1.0,
    n_slices=64,
    n_views=720,
)

# ---- Dual-energy source: airport DECT 80/160 kVp ----------------------------
xray_source_specs = dict(
    num_spectra=2,
    kVp=160,
    spectra=[os.path.join(SPECTRA_DIR, 'airport_spectrum_160kV.txt'),
             os.path.join(SPECTRA_DIR, 'airport_spectrum_80kV.txt')],
    dosage=[5e6, 4e6],
)

# ---- Five-cylinder phantom (1 row × 5 columns) ------------------------------
# Ordered left-to-right: water, RDX, 30% H₂O₂, TATP, RDX/water slurry.
# block_size=30 → 30 mm diameter (3 cm); gap=15 → 15 mm between cylinders.
phantom_materials = ['water', 'rdx', 'h2o2_30', 'tatp', 'rdx_water']

bag_creator_args = dict(
    phantom_mode=True,
    phantom_materials=phantom_materials,
    phantom_grid=(1, 5),        # 1 row, 5 columns
    phantom_block_size=30,      # 30 voxels = 30 mm at 1 mm/voxel
    phantom_gap=15,             # 15 mm inter-cylinder spacing
    material_list=phantom_materials,
    liquid_list=[],
    material_pdf=[1.0 / len(phantom_materials)] * len(phantom_materials),
    liquid_pdf=[],
    prevent_overlap=False,
)

# ---- Decomposition: CDM on GPU ----------------------------------------------
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
params['compress_dicom']    = True
