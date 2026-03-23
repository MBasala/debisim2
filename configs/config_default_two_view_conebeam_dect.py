# -----------------------------------------------------------------------------
"""
Default configuration file for:
    - cone-beam scanner geometry
    - two-view x-ray projection scanning setup
    - simulation of two-view 2d baggage x-ray projections
"""

# -----------------------------------------------------------------------------

__author__    = "Ankit Manerikar"
__copyright__ = "Copyright (C) 2023, Robot Vision Lab"
__date__      = "6th May, 2023"
__credits__   = ["Ankit Manerikar", "Fangda Li"]
__license__   = "Public Domain"
__version__   = "2.1.0"
__maintainer__= ["Ankit Manerikar", "Fangda Li"]
__email__     = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__    = "Prototype"

# -----------------------------------------------------------------------------

from lib.__init__ import *
from lib.misc.stl_loader import discover_pool
from lib.forward_model.scanner_template import ScannerTemplate,\
                                               default_two_view_conebeam

# -----------------------------------------------------------------------------
# Step 1: Specify dataset parameter:

bags_to_create = range(1, 10)                  # Number of bags to create
# simulation directory
sim_dir        = 'results/example_default_two_view_conebeam/'
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 2: Specify Scanner Model using scanner_template.py

scanner_mdl = ScannerTemplate(geometry='cone',
                              scan='circular',
                              machine_dict=default_two_view_conebeam.machine_geometry,
                              recon='fbp',
                              recon_dict=default_two_view_conebeam.recon_params,
                              pscale=1.0)

scanner_mdl.set_recon_geometry()
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
                         kVp=160,
                         spectra=[os.path.join(SPECTRA_DIR,
                                               'example_spectrum_160kV_1.txt'),
                                  os.path.join(SPECTRA_DIR,
                                               'example_spectrum_160kV_2.txt')
                                  ],
    dosage=[2.2752e7]*2
)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
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
            spawn_prob=0.5,         # 50% chance firearms appear in a bag
            count_range=(0, 2),
            materials=['Fe', 'Al'],
            material_pdf=[0.7, 0.3],
        ),
        sharp_objects=dict(
            pool=discover_pool(STL_SHARP_DIR),
            spawn_prob=0.5,         # 50% chance sharp objects appear
            count_range=(0, 2),
            materials=['Fe', 'Ti'],
            material_pdf=[0.6, 0.4],
        ),
        explosives=dict(
            pool=discover_pool(STL_EXPLOSIVES_DIR),
            spawn_prob=0.3,         # 30% chance explosives appear
            count_range=(0, 1),
            materials=['ethanol', 'acetal', 'acrylic'],
            material_pdf=[0.4, 0.3, 0.3],
        ),
        other=dict(
            pool=discover_pool(STL_OTHER_THREATS_DIR),
            spawn_prob=0.3,         # 30% chance other threats appear
            count_range=(0, 1),
            materials=['acetal', 'bakelite'],
            material_pdf=[0.5, 0.5],
        ),
    ),
    fillers=dict(
        pool=discover_pool(STL_FILLERS_DIR),
        count_range=(5, 15),
        materials=mlist,
        material_pdf=material_pdf,
    ),
    liquid_containers=dict(
        pool=discover_pool(STL_LIQUID_CONTAINERS_DIR),
        count_range=(0, 3),
        container_materials=['pyrex', 'polyethylene', 'acrylic'],
        container_material_pdf=[0.4, 0.3, 0.3],
        liquid_materials=['water'],
        liquid_material_pdf=[1.0],
    ),
    # mm_per_voxel: real-world size of each scene voxel.
    # STL files are assumed to be modelled in mm; their native dimensions
    # are preserved in the scene grid.
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
    sheet_dim_list=range(2, 9),  # range of sheet thicknesses
    # -------------------------------------------------------------------------
    # object shape specifications
    dim_range=(20,70),                   # min-max dims of simulated object
    number_of_objects=range(30,40),     # number of objects in each bag
    custom_objects=custom_objects, # if custom objects are to be specified
    custom_obj_prob=0.75,
    # -------------------------------------------------------------------------
    # specifications for metals / target objects
    metal_dict={'metal_amt':  1e2, 'metal_size': (3,5)},
    target_dict={'num_range': (1,3), 'is_liquid': False},
    # -------------------------------------------------------------------------
    # STL object pool configuration
    stl_pool_config=stl_pool_config,
    # overlap prevention — when True, objects are laterally shifted during
    # placement so that most objects do not overlap each other
    prevent_overlap=True,
    # stl_only — when True, only STL pool objects are spawned (threats,
    # fillers, liquid containers); random primitive shapes (ellipsoids,
    # boxes, cylinders, cones, sheets, custom meshes) are disabled
    stl_only=False,
    # -------------------------------------------------------------------------
)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 4 Specify the Dual Energy Decomposition Method

decomp_method = 'none' # constrained decomposition method for DECT

# default values - use gpus for faster 3d image processing
cdm_args = dict(cdm_solver='gpu',
                cdm_type='cpd',     # Compton-PE basis - default
                projector='cpu',
                init_val=(0.1, 0.1))

fwd_mdl_args = dict(
            add_poisson_noise=True,
            add_system_noise=True,
            system_gain=0.0025
        )

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
params['images_to_save']    = ['gt']
params['decomposer']        = decomp_method
params['slicewise']         = False # always False for CBCT
params['fwd_mdl_args']      = fwd_mdl_args
params['dicom_output']      = True
# -----------------------------------------------------------------------------
