# -----------------------------------------------------------------------------
"""
Default configuration file for:
    - a fan-beam parallel scanner geometry
    - dual energy CT setup
    - simulation of a 3D volumetric baggage image
"""
# -----------------------------------------------------------------------------

__author__    = "Ankit Manerikar"
__copyright__ = "Copyright (C) 2023, Robot Vision Lab"
__date__      = "6th April, 2023"
__credits__   = ["Ankit Manerikar", "Fangda Li"]
__license__   = "Public Domain"
__version__   = "2.0.0"
__maintainer__= ["Ankit Manerikar", "Fangda Li"]
__email__     = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__    = "Prototype"

# -----------------------------------------------------------------------------

from lib.__init__ import *
from lib.misc.stl_loader import discover_pool
from lib.forward_model.scanner_template import ScannerTemplate,\
                                               default_scanner_parallel,\
                                               create_parallel_scanner

# -----------------------------------------------------------------------------
# Step 1: Specify dataset parameter:

bags_to_create = range(1, 1)                  # Number of bags to create
sim_dir        = 'results/example_parallelbeam_3d_dect/' # simulation directory
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 2: Specify Scanner Model
#
# Two physical parameters control the scanning geometry:
#   gantry_diameter_mm  — the reconstruction field-of-view diameter (mm)
#   pixel_size_mm       — the reconstructed voxel size (mm)
#
# Everything else (image_dims, det_spacing, det_col_count) is derived.
#
# Option 2 (current): 1024 mm FOV at 2 mm/voxel → 512×512 image
#   Fits objects up to ~700 mm (e.g. M4 carbine at 838 mm diagonally).
#   Same render time as original 512×512 setup.
#
# Option 1 (high-res): 1024 mm FOV at 1 mm/voxel → 1024×1024 image
#   Full 1 mm resolution, but ~4× slower rendering.
#   To switch: set pixel_size_mm=1.0

scanner_mdl = create_parallel_scanner(
    gantry_diameter_mm=1024,    # real security belt FOV ~650-1000 mm
    pixel_size_mm=1.0,          # 2 mm/voxel (change to 1.0 for high-res)
    n_slices=512,
)

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 3: Specify X-ray Source Model

# The X-ray Source Model is specified by a dictionary with the following
# key-value pairs:
# num_spectra  - No of X-ray sources/spectra
# kVp          - peak kV voltage for the X-ray source(s)
# spectra      - file paths for the each of the X-ray spectra.
#                The spectrum files are .txt files containing a N x 2
#                array with the keV values in the first column and
#                normalized photon distribution in the 2nd column.
#                See /include/spectra/ for examples to create your own
#                spectrum file.
# dosage       - dosage count for each of the sources

xray_source_specs = dict(num_spectra=2,
                         kVp=130,
                         spectra=[os.path.join(SPECTRA_DIR,
                                               'example_spectrum_130kV.txt'),
                                  os.path.join(SPECTRA_DIR,
                                               'example_spectrum_95kV.txt')
                                  ],
    dosage=[2e5, 1.85e5]
)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 4: Specify the arguments the BaggageCreator3D() Arguments - these
# arguments decide the nature of objects that can be spawned in the
# simulated bags.

# The list contains the list of materials that will be assigned to the
# objects in the bag - the material assignment is random but liquids need
# to be specified separately if liquid filled containers are to be spawned.

mlist    = ['ethanol',                                       # organic
            'Al', 'Ti', 'Fe',                                # metals
            'bakelite', 'pyrex','acrylic', 'Si',             # glass/
                                                             # ceramics
            'polyethylene',  'pvc', 'polystyrene', 'acetal', # plastics
            'neoprene',                                      # rubber
            'nylon6', 'teflon',                              # cloth
            ]
lqd_list = ['water', 'H2O2']                                 # liquids

# material selection probabilities
material_pdf = [0.3] + [0.05/3]*3 + [0.65/11]*11
liquid_pdf = [1/2., 1/2.]

# using custom shapes other than fixed geometries
custom_objects = [os.path.join(CUSTOM_SHAPES_DIR, s)
                  for s in os.listdir(CUSTOM_SHAPES_DIR)]

# STL object pool configuration ------------------------------------------------
stl_pool_config = dict(
    use_stl_bag=True,               # always use a random STL bag boundary
    bags=dict(
        pool=discover_pool(STL_BAGS_DIR),
        materials=['neoprene', 'nylon6'],
        material_pdf=[0.5, 0.5],
    ),
    threats=dict(
        firearms=dict(
            pool=discover_pool(STL_FIREARMS_DIR),
            spawn_prob=1.0,       # 100% chance firearms appear in a bag
            count_range=(2, 2),   # always exactly 2 items
            materials=['Fe', 'Al'],
            material_pdf=[0.7, 0.3],
        ),
        sharp_objects=dict(
            pool=discover_pool(STL_SHARP_DIR),
            spawn_prob=0.0,         # 50% chance sharp objects appear
            count_range=(0, 2),
            materials=['Fe', 'Ti', 'Al'],
            material_pdf=[0.6, 0.2, 0.2],
        ),
        explosives=dict(
            pool=discover_pool(STL_EXPLOSIVES_DIR),
            spawn_prob=0.0,         # 30% chance explosives appear
            count_range=(0, 1),
            materials=['ethanol', 'acetal', 'acrylic'],
            material_pdf=[0.4, 0.3, 0.3],
        ),
        other=dict(
            pool=discover_pool(STL_OTHER_THREATS_DIR),
            spawn_prob=0.0,         # 30% chance other threats appear
            count_range=(0, 1),
            materials=['acetal', 'bakelite'],
            material_pdf=[0.5, 0.5],
        ),
    ),
    fillers=dict(
        pool=discover_pool(STL_FILLERS_DIR),
        count_range=(0, 0),
        materials=mlist,
        material_pdf=material_pdf,
    ),
    liquid_containers=dict(
        pool=discover_pool(STL_LIQUID_CONTAINERS_DIR),
        count_range=(0, 0),
        container_materials=['pyrex', 'polyethylene', 'acrylic'],
        container_material_pdf=[0.4, 0.3, 0.3],
        liquid_materials=['water'],
        liquid_material_pdf=[1.0],
    ),
    # mm_per_voxel: real-world size of each scene voxel.
    # Must match pixel_size_mm from the scanner above.
    # 1024mm FOV / 512px = 2.0 mm/voxel
    mm_per_voxel=1.0,
    # mesh_units: unit system of the STL files.
    # Most 3D modelling tools export in metres by default.
    # Supported: 'mm', 'cm', 'in'/'inches', 'm'/'meters'
    mesh_units='m',
)
# ------------------------------------------------------------------------------

bag_creator_args = dict(
    # list of materials/liquids to simulate -----------------------------------
    material_list=mlist,
    liquid_list=lqd_list,
    # material selection probabilities - specify for each material ------------
    material_pdf=material_pdf,
    liquid_pdf=liquid_pdf,
    # params for deformable sheets/liquid-filled containers -------------------
    spawn_sheets=True,
    spawn_liquids=True,
    sheet_prob=0.2,       # probability of spawning a deformable sheet
    lqd_prob=0.3,         # probability of spawning a liquid-filled container
    sheet_dim_list=range(2, 10),  # range of sheet thicknesses
    # -------------------------------------------------------------------------
    # object shape specifications
    dim_range=(20,70),                   # min-max dims of simulated object
    number_of_objects=range(30, 40),     # number of objects in each bag
    custom_objects=custom_objects, # if custom objects are to be specified
    custom_obj_prob=0.3,
    # -------------------------------------------------------------------------
    # specifications for metals / target objects
    metal_dict={'metal_amt':  1e2, 'metal_size': (3,5)},
    target_dict={'num_range': (1,3), 'is_liquid': False},
    # -------------------------------------------------------------------------
    # STL object pool configuration
    stl_pool_config=stl_pool_config,
    # -------------------------------------------------------------------------
    # overlap prevention — when True, objects are laterally shifted during
    # placement so that most objects do not overlap each other
    prevent_overlap=True,
    # stl_only — when True, only STL pool objects are spawned (threats,
    # fillers, liquid containers); random primitive shapes (ellipsoids,
    # boxes, cylinders, cones, sheets, custom meshes) are disabled
    stl_only=True,
)
# -----------------------------------------------------------------------------
# Step 4 Specify the Dual Energy Decomposition Method

decomp_method = 'cdm' # constrained decomposition method for DECT

# default values - use gpus for faster 3d image processing
cdm_args = dict(cdm_solver='gpu',
                cdm_type='cpd',     # Compton-PE basis - default
                projector='gpu',
                init_val=(0, 0))
# -----------------------------------------------------------------------------

# params to feed to the debisim pipeline
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
