# RunPod threat detection demo configuration.
#
# Produces a compressed DICOM archive containing:
#   - Dual-energy CT series (80/160 kVp, airport-grade dosage)
#   - RT-Struct with per-object ROI annotations (material, Z_eff, density)
#   - ROI manifest CSV
#
# Objects:
#   - Random STL bag boundary
#   - STL threat objects (firearms, sharp objects, explosives)
#   - STL filler objects (clothing, electronics, etc.)
#   - Optional liquid containers
#   - No primitive shapes (stl_only=True)
#
# Output: results/<sim_dir>/simulation_XXX/images/dicom.zip
#
# Usage:
#   python run_dataset_generator.py \
#       --config configs/config_runpod_threat_demo.py \
#       --sim_dir results/threat_demo/ \
#       --num_bags 1 \
#       --compress_dicom

from lib.__init__ import *
from lib.forward_model.scanner_template import create_parallel_scanner
from lib.misc.stl_loader import discover_pool

# ---- Dataset ----------------------------------------------------------------
bags_to_create = range(1, 2)
sim_dir        = 'results/threat_demo/'

# ---- Scanner: airport-realistic parallel beam --------------------------------
scanner_mdl = create_parallel_scanner(
    gantry_diameter_mm=1024,    # standard airport EDS FOV
    pixel_size_mm=1.0,          # 1mm resolution
    n_slices=350,               # ~35cm z-extent (typical carry-on height)
    n_views=720,                # full angular sampling
)

# ---- Dual-energy source: airport DECT ----------------------------------------
xray_source_specs = dict(
    num_spectra=2,
    kVp=160,
    spectra=[os.path.join(SPECTRA_DIR, 'airport_spectrum_160kV.txt'),
             os.path.join(SPECTRA_DIR, 'airport_spectrum_80kV.txt')],
    dosage=[5e6, 4e6],         # airport-grade (no patient dose constraint)
)

# ---- Material pools ----------------------------------------------------------
# Common baggage materials
mlist = [
    'polyethylene', 'polystyrene', 'nylon6', 'acrylic', 'acetal',
    'polyester', 'pvc', 'bakelite', 'neoprene', 'C',
]
material_pdf = [1.0 / len(mlist)] * len(mlist)

# Liquids that may appear in containers
lqd_list = ['water']
liquid_pdf = [1.0]

# ---- STL object pools --------------------------------------------------------
stl_pool_config = dict(
    use_stl_bag=True,
    bags=dict(
        pool=discover_pool(STL_BAGS_DIR),
        materials=['neoprene', 'nylon6', 'polyester'],
        material_pdf=[0.4, 0.3, 0.3],
    ),
    threats=dict(
        # === TUNE THESE PROBABILITIES FOR YOUR DEMO ===
        firearms=dict(
            pool=discover_pool(STL_FIREARMS_DIR),
            spawn_prob=1,
            count_range=(1, 2),
            materials=['4150crmov', 'Ti'],
            material_pdf=[0.8, 0.2],
        ),
        sharp_objects=dict(
            pool=discover_pool(STL_SHARP_DIR),
            spawn_prob=1,
            count_range=(1, 1),
            materials=['8cr13mov', 'Ti'],
            material_pdf=[0.8, 0.2],
        ),
        explosives=dict(
            pool=discover_pool(STL_EXPLOSIVES_DIR),
            spawn_prob=1,
            count_range=(1, 1),
            materials=['rdx'],
            material_pdf=[1.0],
        ),
        other=dict(
            pool=discover_pool(STL_OTHER_THREATS_DIR),
            spawn_prob=0.0,
            count_range=(0, 0),
            materials=['acetal', 'bakelite', 'pvc'],
            material_pdf=[0.4, 0.3, 0.3],
        ),
    ),
    fillers=dict(
        pool=discover_pool(STL_FILLERS_DIR),
        count_range=(3, 5),            # 5-15 random filler objects
        materials=mlist,
        material_pdf=material_pdf,
    ),
    liquid_containers=dict(
        pool=discover_pool(STL_LIQUID_CONTAINERS_DIR),
        count_range=(0, 2),
        container_materials=['pyrex', 'polyethylene', 'acrylic'],
        container_material_pdf=[0.4, 0.3, 0.3],
        liquid_materials=lqd_list,
        liquid_material_pdf=liquid_pdf,
    ),
    mm_per_voxel=1.0,                   # matches scanner pixel_size_mm
    mesh_units='m',                      # STL files are in meters
)

# ---- Bag creation args -------------------------------------------------------
bag_creator_args = dict(
    stl_pool_config=stl_pool_config,
    stl_only=True,                       # STL objects only, no primitives
    material_list=mlist,
    liquid_list=lqd_list,
    material_pdf=material_pdf,
    liquid_pdf=liquid_pdf,
    spawn_sheets=False,                  # no random sheet objects
    spawn_liquids=True,                  # allow liquid containers from pool
    sheet_prob=0.0,
    lqd_prob=0.3,
    prevent_overlap=True,                # physical placement rules
)

# ---- Decomposition -----------------------------------------------------------
decomp_method = 'cdm'
cdm_args = dict(
    cdm_solver='gpu',
    cdm_type='cpd',
    projector='gpu',
    init_val=(0.1, 0.1),
)

# ---- Pipeline params ---------------------------------------------------------
params = dict()
params['num_bags']          = bags_to_create
params['sim_dir']           = sim_dir
params['scanner']           = scanner_mdl
params['xray_src_mdl']      = xray_source_specs
params['bag_creator_args']  = bag_creator_args
params['save_sino']         = False      # skip sinogram save (saves disk)
params['basis_fn']          = None
params['decomposer_args']   = cdm_args
params['recon_args']        = None
params['images_to_save']    = ['gt', 'lac_1', 'lac_2', 'compton', 'pe', 'zeff']
params['decomposer']        = decomp_method
params['slicewise']         = False
params['dicom_output']      = True       # generate DICOM CT + RT-Struct
params['compress_dicom']    = True       # compress to dicom.zip
