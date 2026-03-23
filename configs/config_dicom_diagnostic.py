# DICOM diagnostic configuration — airport-realistic DECT settings.
#
# Uses 80/160 kVp dual-energy spectra with airport-grade dosage (5e6 photons)
# to validate:
#   1. Image opens without errors in DICOM viewer
#   2. No concentric ring artifacts (Ram-Lak filter regression)
#   3. Material blocks visible with correct contrast ordering
#   4. Window/level adjustable (HU values in expected range)
#   5. Slice navigation works across all 64 slices
#   6. RT-Struct ROI contours align with material blocks
#   7. High-Z metals resolvable (no photon starvation)
#
# Block sizes are physically constrained: metals use thin inserts
# (≤3mm for Cu/Zn, ≤5mm for Fe, ≤10mm for Ti) to stay within the
# scanner's imaging capability at this kVp and dose.

from lib.__init__ import *
from lib.forward_model.scanner_template import create_parallel_scanner

# ---- Dataset ----------------------------------------------------------------
bags_to_create = range(1, 2)
sim_dir        = 'results/dicom_diagnostic/'

# ---- Scanner: small, fast ---------------------------------------------------
scanner_mdl = create_parallel_scanner(
    gantry_diameter_mm=512,     # small FOV -> 256 det columns at 2 mm
    pixel_size_mm=2.0,          # coarse resolution for speed
    n_slices=64,                # minimal z-extent
    n_views=720,                # sufficient angular sampling for 256 pixels
)

# ---- Dual-energy source: airport DECT --------------------------------------
# 80/160 kVp is standard for airport EDS (explosive detection systems).
# Dosage of 5e6 photons/detector is realistic for airport baggage scanners
# which have no patient dose constraint.
xray_source_specs = dict(
    num_spectra=2,
    kVp=160,
    spectra=[os.path.join(SPECTRA_DIR, 'airport_spectrum_160kV.txt'),
             os.path.join(SPECTRA_DIR, 'airport_spectrum_80kV.txt')],
    dosage=[5e6, 4e6],
)

# ---- 5x5 calibration phantom (deterministic, fast) --------------------------
# Materials span the full Z/density range to stress-test windowing.
# Organics/plastics use standard 20-voxel blocks; dense metals use thinner
# blocks to avoid photon starvation (solid Cu is opaque beyond ~3.5mm at 160kVp).
phantom_materials = [
    'air',    'water',  'ethanol',      'polyethylene', 'polystyrene',  # low-Z
    'acrylic', 'nylon6', 'acetal',      'bakelite',     'neoprene',     # mid-Z plastics
    'pyrex',  'salt',   'bone',         'teflon',       'polyester',    # mid-high Z
    'Si',     'Al',     'Ti',           'Fe',           'Cu',           # metals
    'Zn',     'C',      'saline035',    'H2O2',         'pvc',          # mixed
]

bag_creator_args = dict(
    phantom_mode=True,
    phantom_materials=phantom_materials,
    phantom_grid=(5, 5),
    phantom_block_size=20,      # 20 voxels = 40mm at 2mm/voxel
    phantom_gap=10,
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
params['compress_dicom']    = False   # set True or use --compress_dicom CLI flag
