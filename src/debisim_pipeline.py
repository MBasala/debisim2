#!/usr/bin/env python

# ------------------------------------------------------------------------------
"""DEBISimPipeline: Class for generating X-ray projection data for single /
                      dual energy CT scanner models in randomized or user-
                      interactive modes.
"""

__author__    = "Ankit Manerikar"
__copyright__ = "Copyright (C) 2020, Robot Vision Lab"
__date__      = "12th January, 2021"
__credits__   = ["Ankit Manerikar", "Fangda Li", "Dr. Avinash Kak"]
__license__   = "Public Domain"
__version__   = "2.0.0"
__maintainer__= ["Ankit Manerikar", "Fangda Li"]
__email__     = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__    = "Prototype"
# ------------------------------------------------------------------------------

import builtins
import os
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy.ndimage as sptx
import torch
import torch.nn as nn
from typing import List, Any, Optional
import astra
from numpy import (array, zeros, zeros_like, ones, arange, loadtxt,
                   log, exp, clip, unique, moveaxis, newaxis, float32, int16,
                   isscalar, copy, save, savez_compressed)
from tqdm import tqdm
from torch.distributions import Poisson, Normal
from skimage.measure import regionprops
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Pipeline dtype templates
#
# These constants define the numeric precision used throughout the pipeline.
# Changing them in one place propagates to all intermediate computations,
# GPU buffers, and CPU arrays.  Output dtypes for DICOM (int16) and external
# library requirements (ASTRA float32, pyGpufit float32) are NOT templated
# — those are fixed by spec and marked inline.
#
# To switch the pipeline to float64 for higher precision (at 2× memory):
#   NP_FLOAT  = np.float64
#   T_FLOAT   = torch.float64
# ---------------------------------------------------------------------------
NP_FLOAT  = np.float32       # numpy compute dtype
NP_INT    = np.int32          # numpy index/label dtype
T_FLOAT   = torch.float32    # torch compute dtype
T_INT     = torch.long        # torch index dtype (must be long for indexing)
# Output dtype for DICOM/FITS storage — fixed by spec, not templated
STORAGE_DTYPE = np.int16

from lib.__init__ import MU_DIR, SCANNER_DIR, SPECTRA_DIR, DEFAULT_SIM_DIR
from lib.bag_generator.baggage_creator_3d import BaggageImage3D, Object3D
from lib.bag_generator.shape_list_handle import ShapeListHandle
from lib.forward_model.mu_database_handler import MuDatabaseHandler
from lib.misc.ctlib import effective_atomic_number
from lib.bag_generator.baggage_creator_2d import BaggageImage2D
from lib.bag_generator.image_voxelizer_3d import ImageVoxelizer3D
from lib.forward_model.scanner_template import ScannerTemplate
from lib.forward_model.scatter_simulator import ScatterSimulator
from lib.decomposer.cdm_decomposer import CDMDecomposer
from lib.misc.util import (save_fits_data, save_fits_data_async, flush_async_io,
                            read_fits_data, get_logger,
                            submit_async_io, MonolithicArchive)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

"""-----------------------------------------------------------------------------
* Module Description:

This module is the central module for the operation of the DEBISim pipeline - 
it runs the four different blocks of the pipeline to produce simulation data. 
The class DEBISimPipeline() can be used to generate projection data for 
a randomized data generation, for user-interactive data generation in the 
DEBISim GUI and from previously saved shape list data such as for phantoms & saved 
simulation directories. The class is explicitly used in the dataset generation 
in the DEBISim script for the randomized/mode gui. 

Usage:

When initialized, the class DEBISimPipeline() creates a simulation directory 
as specified by the  input argument and creates the following item within the 
directory:

- ground_truth/ - directory containing the ground truth images - these include 
                  gt_label_image, gt_compton, gt_pe_image, gt_zeff_image, 
                  gt_lac_1_image, gt_lac_2_image, gt_hu_1_image, 
                  gt_hu_2_image. The options for which images need to be saved 
                  can be specified while running the program. All images are 
                  saved as compressed .fits.gz files and can be read using 
                  util.read_fits_data files.
- sinograms/ -    This directory is reserved for saving the polychromatic 
                  projection data generating the module. The preferred 
                  nomenclature for the sinograms is sino_%i.fits.gz, where 
                  i = 1, 2, ... corresponding to the respective spectra of 
                  the projection.
- images/ -       This subdirectory is reserved for saving the reconstructed 
                  single-energy/multi-energy CT images from the projection 
                  data. The reconstructed images are saved as DICOM/DICOS 
                  files. Any post-processing output such as for MAR or 
                  Segmentation can be saved in this subdirectory. The preferred 
                  nomenclature for the image files is recon_image_%s.fits.gz 
                  (where %s - 1,2,...n - for the CT images, c - for compton, 
                  p - for PE and z - for effective atomic number.)
- sl_metadata.pyc This pickle file contains the Shape List for the simulated 
                  ground truth. See ShapeListHandle() for more details.

The ground truth data for the simulator instance can be created either in a 
randomized mode or a user specified mode. In the randomized mode, the 
simulator calls in the BaggageCreator3D module to spawn a randomized baggage 
simulation with objects randomly selected, placed and oriented as specified 
by the baggage creator arguments. In the user specified mode, the objects to 
be spawned in the baggage simulation are specified by a shape list provided 
as input. The parameters within the Shape List can be fed in using a 
Simulator GUI or manually by creating an SL dictionary for each object. 

An example for the use of the module in a randomized mode is given below. To 
create the instance, one must first specify (i) a scanner model using the
ScannerTemplate module, (ii) a Python dictionary specifying the X-ray source 
model, i.e., the spectrum pdfs, the dosage and peak tube voltages (see below) 
and (iii) the argument for creating a randomized bag using BaggageCreator3D
module - this involves selecting of materials, object dimensions and the number 
of objects. The ground truth image can then be generated using the 
self.create_random_simulation_instance() method. The projection data is 
created using the self.run_fwd_model(). Projection data generation includes 
options for adding Poisson noise and Gaussian shot noise to create noisy 
projections. The code for the example is as follows:


Methods:

__init__                                - Constructor 
create_random_simulation_instance       - Creates a random simulation instance 
                                          using BaggageCreator3D 
create_simulation_from_sl_file          - Creates simulaiton instance by reading 
                                          in a shape list
generate_polychromatic_ct_projection    - Generate polychromatic projection for 
                                          a single spectrum
run_fwd_model   - Generate projection scanner for the 
                                          entire scanner/source model
save_dect_ground_truth_images           - Save the ground truth label/coefficient 
                                          images

Attributes:

f_loc                                   - dictionary containing default file 
                                          locations
gt_image_3d                             - GT label image
image_shape_3d                          - dimensions of the volumetric ground 
                                          truth image
keV_range                               - the keV range for the Xray spectrum
material_curve                          - 2D array of attenuation curves for 
                                          all the materials in the image
maxkV                                   - maximum keV value encountered
mu                                      - instance of the MuDatabaseHandler 
                                          used in the image. 
reconstruction_geometry                 - reconstruction geometry read from 
                                          scanner model
scale                                   - image scale
scanner_geometry                        - scanner geometry read from scanner 
                                          model
scanner                                 - scanner model as an instance of 
                                          ScannerTemplate()                        
sf_obj_list                             - Shape List for the simulation 
                                          instance
slh                                     - ShapeListHandle() object for the 
                                          class
xray_source_model                       - dictionary of Xray source 
                                          specifications: {'num_spectra': number
                                          of spectra, 'dosage': list of dosage 
                                          counts for each spectra, 'spectrum':
                                          list of path for each spectrum file 
                                          (For format, see ./include/spectra/),
                                          'kVp': list of tube voltages for 
                                          each spectrum}
zwidth                                  - z-axis width of the image in mm.
                                          (Number of slices in unit mm)
-------------------------------------------------------------------------------
"""

default_file_locations = dict(
    simulation_dir=DEFAULT_SIM_DIR,
    scanner_dir=SCANNER_DIR,
    mass_attn_dir=MU_DIR,
    spectra_dir=SPECTRA_DIR,
    image_dir='images/',
    sino_dir='projections/',
    gt_dir='ground_truth/',
    sino_file='sino_%i.fits.gz',
    img_file='recon_image_%i.fits.gz',
    gt_image='gt_label_image.fits.gz',
    sl_metadata='sl_metadata.pyc'
)


DEFAULT_FWD_MODEL_ARGS = dict(add_poisson_noise=True,
                              add_system_noise=True,
                              system_gain=1)


class DEBISimPipeline(object):

    # Methods -----------------------------------------------------------------

    def __init__(
            self,
            sim_path,
            scanner_model,
            xray_source_model,
            mu_handler=None,
            debug=False,
            logfile=None,
            compress_data=False,
            monolithic_output=False,
            compression_threads=0
    ):
        """
        ------------------------------------------------------------------------
        Constructor for the DEBISimPipeline Class.

        :param sim_path: The path to the simulation directory. When the simulation
                         is complete, this directory will be populated with
                         the subdirectories images/, sinograms/, and
                         ground_truth/ containing the related data.
                         
        :param scanner_model: An instance of the ScannerTemplate where the 
                              scanner specifications are given.
        
        :param xray_source_model: Dictionary specifying the X-ray source
                             specifications. See Module Description for details

        :param mu_handler:    A MuDatabaseHandler object (optional)
        :param logfile:       log file for simulation
        :param debug:         set if in debug mode
        :param compress_data: set to compress saved FITS data
        ------------------------------------------------------------------------
        """

        # Select default values for the f_loc and scanner_specs
        self.f_loc = default_file_locations.copy()
        self.DECOMPOSER_FLAG = False

        self.scanner                 = scanner_model
        self.scanner_geometry        = scanner_model.machine_geometry.copy()
        self.reconstruction_geometry = scanner_model.recon_geometry.copy()
        self.xray_source_model       = xray_source_model.copy()
        self.zwidth                  = self.scanner.recon_params['image_dims'][2]
        self.debug                   = debug
        self.logfile                 = logfile
        self.compress_data           = compress_data

        self.maxkV = self.xray_source_model['kVp']  \
                     if   isscalar(self.xray_source_model['kVp']) \
                     else max(self.xray_source_model['kVp'])

        # Assign and create a simulation directory for the CT simulation
        self.f_loc['simulation_dir'] = sim_path

        os.makedirs(self.f_loc['simulation_dir'], exist_ok=True)

        # Assign absolute paths to all simulation files
        for sim_file in ['image_dir', 'sino_dir', 'gt_dir']:
            self.f_loc[sim_file] = os.path.join(self.f_loc['simulation_dir'],
                                                self.f_loc[sim_file])

            os.makedirs(self.f_loc[sim_file], exist_ok=True)

        self.f_loc['gt_image'] = os.path.join(self.f_loc['gt_dir'],
                                            self.f_loc['gt_image'])
        self.f_loc['sl_metadata'] = os.path.join(self.f_loc['simulation_dir'],
                                            self.f_loc['sl_metadata'])

        self.f_loc['sino_file'] = os.path.join(self.f_loc['sino_dir'],
                                            self.f_loc['sino_file'])

        # ---------------------------------------------------------------------

        self.image_shape_3d = tuple(
            int(d) for d in self.scanner.recon_params['image_dims']
        )

        self.slh = ShapeListHandle()
        self.mu = MuDatabaseHandler(self.debug, self.logfile) \
                  if mu_handler is None else mu_handler

        self.keV_range = arange(10, self.maxkV+1)

        spectra = [loadtxt(spec)[:self.maxkV-10,1]
                   for spec in self.xray_source_model['spectra']]

        self.mu.calculate_lac_hu_values('water', spectra)

        # LAC unit scaling for the forward model energy loop.
        #
        # The mu database stores mass attenuation coefficients in cm²/g.
        # Multiplying by density gives LAC in cm⁻¹.
        #
        # self.scale is ALWAYS 0.1 (cm⁻¹ → mm⁻¹) in the mu_curve that
        # feeds the energy loop.  This keeps the projection values in a
        # range where exp(-proj) produces meaningful photon counts for the
        # Poisson noise model (system_gain=2.5e-3 was calibrated for this).
        #
        # img_scale (applied after FBP) converts the reconstruction back
        # to physical LAC units for HU conversion.  Their product must
        # produce correct LAC:
        #
        #   mu_curve = mu(cm²/g) × density(g/cm³) × 0.1
        #   recon_lac = FBP(sinogram) × img_scale
        #   HU = (recon_lac - mu_w) / mu_w × 1000
        #
        # For img_scale=1.0 (FBP_CUDA), self.scale=0.1 means the FBP
        # output is in mm⁻¹.  The sinogram post-processing then scales
        # by (1 / self.scale) to convert back to cm⁻¹ before saving.
        #
        self.scale = 0.1  # always cm→mm for noise model compatibility

        # Beam hardening / metal artifact correction flags.
        # Set via config or overridden before run_reconstructor().
        self.apply_bhc = True   # water-based BHC (sinogram domain)
        self.apply_mar = False  # NMAR metal artifact reduction

        # Monolithic archive mode — all outputs stream into a single
        # .tar.gz via a subprocess with its own GIL.  Per-file FITS
        # compression is disabled (the archive handles compression).
        self.monolithic_output = monolithic_output
        self.archive = None
        if monolithic_output:
            self.compress_data = False  # archive compresses everything
            archive_path = self.f_loc['simulation_dir'].rstrip('/\\') + '.tar.gz'
            self.archive = MonolithicArchive(
                archive_path,
                compression_threads=compression_threads,
                ram_limit_fraction=0.5,
            )

        if self.logfile is None:
            self.logfile = os.path.join(self.f_loc['simulation_dir'],
                                        'debisim_bag.log')
            if not os.path.exists(self.logfile):
                open(self.logfile, 'w+').close()

        self.logger = get_logger('DEBISIM', self.logfile)

        if self.archive is not None:
            self.archive._logger = self.logger

        header = ['CT Specifications', '']
        print_table = []

        print_table.append(['Initialization Time',
                            time.strftime('%m-%d-%Y %H:%M:%S',
                                          time.localtime())])
        print_table.append(['Simulation Directory',
                            self.f_loc['simulation_dir']])
        print_table.append(['Image Dimensions', self.image_shape_3d])
        print_table.append(['CT Scanner', self.scanner_geometry['scanner_name']])
        print_table.append(['Projection Dims. (views, rows, cols)',
                            [self.reconstruction_geometry['n_views'],
                             self.scanner_geometry['det_row_count'],
                             self.scanner_geometry['det_col_count']]
                            ])

        self.logger.info('\n'+tabulate(print_table, header, tablefmt='psql'))
        self.logger.info('\n')
    # --------------------------------------------------------------------------

    def create_random_simulation_instance(self,
                                          baggage_creator_args,
                                          prior_image=None,
                                          prior_list=None,
                                          save_images=['gt'],
                                          slicewise=False,
                                          template=2
                                          ):
        """
        ------------------------------------------------------------------------
        Create a random simulation phantom from the randomized BaggageCreator3D. 
        The functions also allows spawning randmized objects over a prior image.

        :param baggage_creator_args: arguments to run the 
                              BaggageCreator3D.create_random_object_list()
                              function. 
        :param prior_image:   Prior image if needs to be included
        :param prior_list:    Shape List of objects in the prior image
        :param save_images:   ground truth images to save (Options: {'gt', 
                              'compton', 'pe', 'zeff', 'lac_1', 'lac_2'})
        :return:
        ------------------------------------------------------------------------
        """

        # run BaggageCreator3D with the input creator args ---------------------
        if prior_list is None:
            prior_list = []
        bag_vol_shape = list(self.image_shape_3d)
        # bag_vol_shape[2] = max(bag_vol_shape[2], 350)

        if not slicewise:
            # BaggageImage3D.gantry_dia controls the bag boundary polygon
            # and ws_bag working volume size.  It expects the same units as
            # img_vol (voxels), but upstream hardcoded offsets assume ~512.
            # Pass gantry_diameter_mm here (matches upstream's assumption
            # that pixel_size=1mm) — the ws_bag is oversized but gets
            # cropped to img_vol at finalization.
            virtual_bag_creator = BaggageImage3D(img_vol=tuple(bag_vol_shape),
                                                 sim_dir=self.f_loc['gt_dir'],
                                                 logfile=self.logfile,
                                                 gantry_dia=int(
                                                     self.scanner_geometry['gantry_diameter']),
                                                 prior_image=prior_image,
                                                 debug=self.debug
                                         )
        else:
            virtual_bag_creator = BaggageImage2D(img_vol=tuple(bag_vol_shape),
                                                 sim_dir=self.f_loc['gt_dir'],
                                                 logfile=self.logfile,
                                                 gantry_dia=int(
                                                     self.scanner_geometry['gantry_diameter']),
                                                 prior_image=prior_image,
                                                 template=template,
                                                 debug=self.debug
                                         )

        # get Object3D list — either calibration phantom or random bag
        _bag_args = dict(baggage_creator_args)
        prevent_overlap = _bag_args.pop('prevent_overlap', False)
        phantom_mode = _bag_args.pop('phantom_mode', False)

        if phantom_mode:
            phantom_materials = _bag_args.pop('phantom_materials')
            phantom_grid = _bag_args.pop('phantom_grid', (5, 5))
            phantom_block_size = _bag_args.pop('phantom_block_size', 30)
            phantom_gap = _bag_args.pop('phantom_gap', 20)

            # Phantom places blocks directly into ws_bag and finalizes
            # virtual_bag — no Object3D / create_baggage_image needed.
            virtual_bag_creator.create_calibration_phantom(
                materials=phantom_materials,
                grid=phantom_grid,
                block_size=phantom_block_size,
                gap=phantom_gap,
            )
        else:
            obj_list = virtual_bag_creator.create_random_object_list(
                                                **_bag_args
                                                )
            # run placement logic
            virtual_bag_creator.create_baggage_image(obj_list,
                                                     save_data=False,
                                                     prevent_overlap=prevent_overlap)
        sf_obj_list = virtual_bag_creator.param_file + prior_list

        # This is your final virtual bag
        self.gt_image_3d = virtual_bag_creator.virtual_bag

        virtual_bag_creator.logger.propagate = False
        # virtual_bag_creator.mu_handler.logger.propagate = False

        # adjust baggage volume to zwidth
        if self.zwidth < self.gt_image_3d.shape[2]:
            zd = (self.gt_image_3d.shape[2] - self.zwidth)//2
            self.gt_image_3d = self.gt_image_3d[:,:,zd:-zd]

        self.gt_image_3d = torch.as_tensor(self.gt_image_3d)

        if virtual_bag_creator.logger.hasHandlers():
            virtual_bag_creator.logger.handlers.clear()

        del virtual_bag_creator.logger
        del virtual_bag_creator
        # ---------------------------------------------------------------------

        self.sf_obj_list = sf_obj_list

        # create compton_image - this is used for forward modeling
        self.compton_image_3d = torch.zeros_like(self.gt_image_3d,
                                                 dtype=T_FLOAT)

        # replace gt label with compton value ---------------------------------
        for sf_obj in sf_obj_list:
            self.compton_image_3d = torch.where(
                self.gt_image_3d==sf_obj['label'],
                torch.Tensor([self.mu.material(sf_obj['material'], 
                                               'compton')]),
                self.compton_image_3d)

            if sf_obj['lqd_flag']:
                self.compton_image_3d = torch.where(
                    self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                    torch.Tensor([self.mu.material(sf_obj['lqd_param']['lqd_material'],
                                                   'compton')]),
                    self.compton_image_3d)
        # ---------------------------------------------------------------------

        # get attenuation curves for each material in the bag -----------------
        spectra = [loadtxt(spec)[:self.maxkV,1]
                   for spec in self.xray_source_model['spectra']]

        self.mu.calculate_lac_hu_values('water', spectra)

        sim_obj_dict = {}
        material_curve = {}

        # iterating through each object
        for k, sf_obj in enumerate(sf_obj_list):

            mu_curve =  zeros(len(self.keV_range))
            self.mu.calculate_lac_hu_values(sf_obj['material'], spectra)
            curr_material = self.mu.material(sf_obj['material'])

            atten_curve =  curr_material['mu']  # original unscaled atten. curve

            # Note: The mu database stores mass attenuation coefficients in
            #       cm²/g.  Multiplying by density converts to LAC (cm⁻¹).
            #       self.scale (fixed at 0.1, see __init__) converts cm⁻¹
            #       to mm⁻¹ for the ASTRA projection geometry.  The inverse
            #       correction (×10) is applied later via sino_correction
            #       in run_reconstructor to recover cm⁻¹ for HU conversion.

            # the array limits for mu_curve and atten_curve are to ensure
            # they are of the same length, mac value is multiplied by density
            # to convert to lac
            mu_curve[:atten_curve.size] = \
                atten_curve[:len(self.keV_range)]*curr_material['density']*self.scale

            # save the mu_curve in obj metadata
            sf_obj['mu_curve'] = mu_curve
            sf_obj['mu_dict']  = curr_material.copy()

            # add the mu curve to material curves dictionary
            material_curve[sf_obj['material']] = mu_curve

            # repeat for liquids
            if sf_obj['lqd_flag']:
                mu_curve = zeros(len(self.keV_range))

                self.mu.calculate_lac_hu_values(sf_obj['lqd_param']['lqd_material'],
                                                spectra)
                curr_material = self.mu.material(sf_obj['lqd_param']['lqd_material'])
                atten_curve = curr_material['mu']

                mu_curve[:atten_curve.size] = \
                    atten_curve[:len(self.keV_range)] * curr_material['density'] * self.scale

                sf_obj['lqd_param']['mu_curve'] = mu_curve
                sf_obj['lqd_param']['mu_dict'] = curr_material.copy()
                material_curve[sf_obj['lqd_param']['lqd_material']] = mu_curve

            sim_obj_dict['%i' % sf_obj['label']] = sf_obj.copy()
        # ---------------------------------------------------------------------

        # save the dictionary of objects as metadata
        self._save_pickle(self.f_loc['sl_metadata'], sim_obj_dict)

        self.logger.info(f"Metadata generated and saved "
                         f"at {self.f_loc['sl_metadata']}")
        # ----------------------------------------------------------------------

        self.logger.info(f"Number of Objects: {len(sf_obj_list)}")
        self.logger.info("Details:")

        print_table = []
        header = ['Label', 'Shape', 'Material', 'Lqd.']

        if not slicewise:
            header = header + ['Center/Base', 'Dim/Apex', 'Rot/Radius']

        for sf_obj in sf_obj_list:

            print_list = [str(sf_obj['label']), sf_obj['shape'],
                          sf_obj['material'], sf_obj['lqd_flag']]

            if sf_obj['shape'] in ['B', 'S', 'T', 'M']:
                ind1, ind2, ind3 = 'center', 'dim', 'rot'
            elif sf_obj['shape'] in ['E']:
                ind1, ind2, ind3 = 'center', 'axes', 'rot'
            elif sf_obj['shape'] in ['Y']:
                ind1, ind2, ind3 = 'base', 'apex', 'radius'
            elif sf_obj['shape'] in ['C']:
                ind1, ind2, ind3 = 'base', 'apex', 'radius1'

            if not slicewise:
                print_list.append('(%i,%i,%i)' % (sf_obj['geom'][ind1][0],
                                                  sf_obj['geom'][ind1][1],
                                                  sf_obj['geom'][ind1][2]))
                print_list.append('(%i,%i,%i)' % (sf_obj['geom'][ind2][0],
                                                  sf_obj['geom'][ind2][1],
                                                  sf_obj['geom'][ind2][2]))

            if sf_obj['shape'] in ['B', 'E', 'S', 'T', 'M']:
                if not slicewise:
                    print_list.append('(%i,%i,%i)' % (sf_obj['geom'][ind3][0],
                                                      sf_obj['geom'][ind3][1],
                                                      sf_obj['geom'][ind3][2]))

            else:
                print_list.append('%i' % (sf_obj['geom'][ind3]))

            print_table.append(print_list)

        self.logger.info("\n"+tabulate(print_table,
                                  headers=header,
                                  tablefmt='grid'))
        # self.logger.info()

        self.logger.info('=' * 40)
        self.material_curve = material_curve

        if 'obj_list' in dir():
            del obj_list
        self.save_dect_ground_truth_images(images=save_images)

        torch.cuda.empty_cache()
    # --------------------------------------------------------------------------

    def create_simulation_from_sl_file(self,
                                       shape_list_file,
                                       gt_image=None,
                                       save_images=['gt']
                                       ):
        """
        ------------------------------------------------------------------------
        Function to initialize simulation by reading a shape list from a
        previously saved shape list. The function can voxelize a ground truth
        image from the shape list as long as it does not contain liquids or
        sheet objects, otherwise, gt_image needs to be provided.

        :param shape_list_file: SL instance or file-path of the SL file.
        :param gt_image:        GT Label image corresponding to the shape file
        :param save_images:     ground truth images to save (Options: {'gt',
                                'compton', 'pe', 'zeff', 'lac_1', 'lac_2'})
        :return:
        ------------------------------------------------------------------------
        """

        if isinstance(shape_list_file, str):
            with open(shape_list_file, 'rb') as f:
                sf_obj_dict = pickle.load(f, encoding='latin1')
                sf_obj_list = [sf_obj_dict['%i'%x]
                               for x in range(1, len(sf_obj_dict.keys())+1)
                               ]
                f.close()
        else:
            sf_obj_list = shape_list_file

        # ----------------------------------------------------------------------

        spectra = [loadtxt(spec)[:self.maxkV,1]
                   for spec in self.xray_source_model['spectra']]

        sim_obj_dict = {}
        material_curve = {}

        # TODO: check load sim with liquid objects
        for k, sf_obj in enumerate(sf_obj_list):
            mu_curve = zeros(len(self.keV_range))

            self.mu.calculate_lac_hu_values(sf_obj['material'], spectra)
            curr_material = self.mu.material(sf_obj['material'])
            atten_curve = curr_material['mu']
            mu_curve[:atten_curve.size] = atten_curve[:len(self.keV_range)] * curr_material['density'] * self.scale
            sf_obj['mu_curve'] = mu_curve
            sf_obj['mu_dict'] = curr_material.copy()
            material_curve[sf_obj['material']] = mu_curve.copy()

            if sf_obj['lqd_flag']:
                mu_curve = zeros(len(self.keV_range))

                self.mu.calculate_lac_hu_values(sf_obj['lqd_param']['lqd_material'],
                                                spectra)
                curr_material = self.mu.material(sf_obj['lqd_param']['lqd_material'])
                atten_curve = curr_material['mu']
                mu_curve[:atten_curve.size] = atten_curve[:len(self.keV_range)] * curr_material['density'] * self.scale
                sf_obj['lqd_param']['mu_curve'] = mu_curve
                sf_obj['lqd_param']['mu_dict'] = curr_material.copy()
                material_curve[sf_obj['lqd_param']['lqd_material']] = mu_curve.copy()

            sim_obj_dict['%i' % sf_obj['label']] = sf_obj.copy()

        self._save_pickle(self.f_loc['sl_metadata'], sim_obj_dict)

        self.logger.info("\nMetadata generated")
        # ----------------------------------------------------------------------

        self.logger.info("\nNumber of Objects: %i"%len(sf_obj_list))
        self.logger.info("Details:")

        print_table = []
        header = ['Label', 'Shape', 'Material', 'Lqd.',
                  'Center/Base', 'Dim/Apex', 'Rot/Radius']

        for sf_obj in sf_obj_list:

            print_list = [str(sf_obj['label']), sf_obj['shape'],
                          sf_obj['material'], sf_obj['lqd_flag']]

            if sf_obj['shape'] in ['B', 'S', 'T', 'M']:
                ind1, ind2, ind3 = 'center', 'dim', 'rot'
            elif sf_obj['shape'] in ['E']:
                ind1, ind2, ind3 = 'center', 'axes', 'rot'
            elif sf_obj['shape'] in ['Y']:
                ind1, ind2, ind3 = 'base', 'apex', 'radius'
            elif sf_obj['shape'] in ['C']:
                ind1, ind2, ind3 = 'base', 'apex', 'radius1'

            print_list.append('(%i,%i,%i)' % (sf_obj['geom'][ind1][0],
                                              sf_obj['geom'][ind1][1],
                                              sf_obj['geom'][ind1][2]))
            print_list.append('(%i,%i,%i)' % (sf_obj['geom'][ind2][0],
                                              sf_obj['geom'][ind2][1],
                                              sf_obj['geom'][ind2][2]))

            if sf_obj['shape'] in ['B', 'E', 'S', 'T', 'M']:
                print_list.append('(%i,%i,%i)' % (sf_obj['geom'][ind3][0],
                                                  sf_obj['geom'][ind3][1],
                                                  sf_obj['geom'][ind3][2]))
            else:
                print_list.append('%i' % (sf_obj['geom'][ind3]))

            print_table.append(print_list)

        self.logger.info("\n"+tabulate(print_table,
                                       headers=header,
                                       tablefmt='grid'))

        self.logger.info('-' * 40+'\n')

        self.sf_obj_list = sf_obj_list
        self.material_curve = material_curve

        torch.cuda.empty_cache()

        if gt_image is None:
            voxelizer = ImageVoxelizer3D(sf_list=self.sf_obj_list,
                                         imgshape=self.image_shape_3d)

            self.compton_image_3d, self.gt_image_3d = \
                voxelizer.voxelize_3d_image()

            del voxelizer
        else:
            self.gt_image_3d = torch.as_tensor(gt_image)

            # Build compton image via LUT instead of per-object torch.where loop
            _max_lbl = int(self.gt_image_3d.max().item()) + 1
            _compton_lut = torch.zeros(_max_lbl, dtype=torch.float,
                                       device=self.gt_image_3d.device)
            for sf_obj in sf_obj_list:
                _compton_lut[sf_obj['label']] = self.mu.material(
                    sf_obj['material'], 'compton')
                if sf_obj.get('lqd_flag') and sf_obj.get('lqd_param'):
                    _compton_lut[sf_obj['lqd_param']['lqd_label']] = \
                        self.mu.material(sf_obj['lqd_param']['lqd_material'],
                                         'compton')

            self.compton_image_3d = _compton_lut[
                self.gt_image_3d.long().clamp(0, _max_lbl - 1)]

        self.save_dect_ground_truth_images(images=save_images)
    # --------------------------------------------------------------------------

    def generate_polychromatic_ct_projection(self,
                                            add_poisson_noise=True,
                                            add_system_noise=True,
                                            system_gain=2.5e-3,
                                            shot_gain=5e-5,
                                            spectrum=1):
        """
        ------------------------------------------------------------------------
        Generate polyenergetic sinogram for the ground truth label image. The
        ground truth image must be generated prior to calling this function. The
        function generates the noisy polychromatic sinogram for the specified
        spectrum in the Xray source model dictionary, self.xray_source_model and
        saves it in the sino/ folder in the simulation directory.

        :param add_poisson_noise:   Set to True if Poisson noise is to be added.
        :param add_system_noise:    Set to True if Gaussian shot noise is to be
                                    added.
        :param system_gain:         Gain for Gaussian shot noise
        :param spectrum:            index of the spectrum as specified in self.
                                    xray_source_model.

        :return
        ------------------------------------------------------------------------
        """

        t0 = time.time()

        i = spectrum
        self.logger.info("Generating Polyenergetic Sinograms "
                         "for Spectrum %i ..."%(i))

        curr_spectrum =  loadtxt(self.xray_source_model['spectra'][i-1])[:, 1]
        curr_pc = self.xray_source_model['dosage'][i-1]

        material_list = unique(list(self.material_curve.keys()))
        pc_sum = 0

        # --- Build label-to-material index map on CPU (numpy) -------------------
        # The ref_image for each keV is assembled via numpy LUT indexing and
        # passed to ASTRA (which runs its own GPU kernels).  Keeping this on
        # CPU means torch doesn't hog VRAM while ASTRA needs it.
        mat_indices = {mat: idx for idx, mat in enumerate(material_list)}

        # max_label must cover ALL labels including liquid labels that may
        # exceed gt_image_3d.max() (e.g. when liquid fill partially failed).
        max_label = int(self.gt_image_3d.max().item()) + 1
        for sf_obj in self.sf_obj_list:
            max_label = builtins.max(max_label, sf_obj['label'] + 1)
            if sf_obj.get('lqd_flag') and sf_obj.get('lqd_param'):
                max_label = builtins.max(
                    max_label, sf_obj['lqd_param']['lqd_label'] + 1)

        _label_to_mat_idx = np.full(max_label, -1, dtype=NP_INT)
        for sf_obj in self.sf_obj_list:
            mat = sf_obj['material']
            if mat in mat_indices:
                _label_to_mat_idx[sf_obj['label']] = mat_indices[mat]
            if sf_obj.get('lqd_flag') and sf_obj.get('lqd_param'):
                lqd_mat = sf_obj['lqd_param']['lqd_material']
                if lqd_mat in mat_indices:
                    _label_to_mat_idx[sf_obj['lqd_param']['lqd_label']] = \
                        mat_indices[lqd_mat]

        # ---- Build LUT + index maps on BOTH CPU and GPU ----------------------
        # CPU RAM is plentiful (64 GB) so we duplicate the index map:
        #   - CPU copy:  feeds ASTRA via np.take (no GPU→CPU transfer per keV)
        #   - GPU copy:  feeds torch post-processing (exp, noise, accumulate)
        # The LUT itself is tiny (n_mats × n_keV ≈ 7 KB).
        _gt_device = self.gt_image_3d.device

        _label_to_mat_idx_gpu = torch.tensor(
            _label_to_mat_idx, dtype=T_INT, device=_gt_device)
        gt_clamped = self.gt_image_3d.long().clamp(0, max_label - 1)
        voxel_mat_idx_gpu = _label_to_mat_idx_gpu[gt_clamped]
        del gt_clamped, _label_to_mat_idx_gpu

        bg_mask_gpu = (voxel_mat_idx_gpu < 0)
        voxel_mat_idx_safe_gpu = voxel_mat_idx_gpu.clamp(min=0)
        del voxel_mat_idx_gpu

        # CPU copies for ASTRA (duplicated — CPU RAM is cheap)
        voxel_mat_idx_safe_cpu = voxel_mat_idx_safe_gpu.cpu().numpy()
        bg_mask_cpu = bg_mask_gpu.cpu().numpy()

        # Pre-build per-keV LAC lookup table on both CPU and GPU
        n_kev = len(self.keV_range[:curr_spectrum.size])
        n_mats = len(material_list)
        # Note: ASTRA requires float32 input regardless of NP_FLOAT.
        # The LUT is built at NP_FLOAT precision, then the ref_image fed to
        # ASTRA is always np.float32 (cast at the projector boundary).
        lac_lut_cpu = np.zeros((n_mats, n_kev), dtype=NP_FLOAT)
        for mat in material_list:
            idx = mat_indices[mat]
            for ki in range(n_kev):
                lac_lut_cpu[idx, ki] = self.material_curve[mat][ki]
        lac_lut_gpu = torch.tensor(lac_lut_cpu, device=_gt_device)

        # Pre-allocate reusable CPU buffer for ASTRA input (avoids alloc per keV)
        ref_image_cpu = np.empty(voxel_mat_idx_safe_cpu.shape, dtype=np.float32)

        # ---- Temporarily move large tensors off GPU --------------------------
        # gt_image_3d and compton_image_3d are ~2 GB each on a 1024×1024
        # volume.  They're no longer needed until after the energy loop.
        # GPU now holds: voxel_mat_idx_safe_gpu (~350 MB) + bg_mask_gpu (~90 MB)
        #              + lac_lut_gpu (~7 KB) + projection_buffer (~130 MB)
        #              ≈ 570 MB, leaving ~11 GB for ASTRA.
        self.gt_image_3d = self.gt_image_3d.cpu()
        self.compton_image_3d = self.compton_image_3d.cpu()
        if hasattr(self, 'pe_image_3d') and torch.is_tensor(self.pe_image_3d):
            self.pe_image_3d = self.pe_image_3d.cpu()
        if hasattr(self, 'zeff_image_3d') and torch.is_tensor(self.zeff_image_3d):
            self.zeff_image_3d = self.zeff_image_3d.cpu()

        # The energy loop uses CPU copies for ASTRA, so free the GPU
        # index/mask tensors too — they were only needed to build the CPU copies.
        del voxel_mat_idx_safe_gpu, bg_mask_gpu, lac_lut_gpu

        # Free torch's cached GPU memory so ASTRA can use the full VRAM.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                _free, _total = torch.cuda.mem_get_info(0)
                astra.set_gpu_index(0, memory=int(_free * 0.9))
                self.logger.info(
                    f"ASTRA GPU memory: {_free/1e9:.1f} GB free of "
                    f"{_total/1e9:.1f} GB total")
            except Exception:
                try:
                    astra.set_gpu_index(0)
                except Exception:
                    pass
        # --------------------------------------------------------------------------

        # Accumulation buffer on GPU — post-processing (exp, noise) runs on GPU.
        proj_shape = (self.scanner_geometry['det_row_count'],
                      self.reconstruction_geometry['n_views'],
                      self.scanner_geometry['det_col_count'])
        projection_buffer_gpu = torch.zeros(proj_shape, dtype=T_FLOAT,
                                            device=_gt_device)

        # ---- Per-keV LUT gather + ASTRA projection ----------------------------
        # Double-buffered pipeline: a prefetch thread prepares the next
        # ref_image on CPU while ASTRA projects the current one on GPU.
        # ASTRA's C extension releases the GIL during CUDA kernel execution,
        # so the CPU LUT gather runs truly concurrently with GPU work.
        vol_shape = voxel_mat_idx_safe_cpu.shape

        # Two buffers — one for ASTRA, one being filled by the prefetch thread
        ref_buf_a = np.empty(vol_shape, dtype=np.float32)
        ref_buf_b = np.empty(vol_shape, dtype=np.float32)

        kev_list = list(self.keV_range[:curr_spectrum.size])
        kev_iter = tqdm(kev_list)

        # Prefetch helper: fills a buffer with the LAC image for keV index k
        def _prefetch(buf, k):
            np.take(lac_lut_cpu[:, k], voxel_mat_idx_safe_cpu, out=buf)
            buf[bg_mask_cpu] = 0.0

        # Single-thread pool reused across all keV steps — avoids spawning
        # 150+ bare threads.  The pool thread stays alive between submissions.
        from concurrent.futures import ThreadPoolExecutor
        prefetch_pool = ThreadPoolExecutor(max_workers=1,
                                           thread_name_prefix='kev_prefetch')

        # Seed the first buffer synchronously
        _prefetch(ref_buf_a, 0)

        for i, e in enumerate(kev_iter):
            # ref_buf_a holds the ready image for this keV step.
            # Start prefetching the NEXT keV into ref_buf_b while ASTRA runs.
            prefetch_fut = None
            if i + 1 < len(kev_list):
                next_k = kev_list[i + 1] - self.keV_range[0]
                prefetch_fut = prefetch_pool.submit(_prefetch, ref_buf_b,
                                                    next_k)

            # --- ASTRA GPU — forward projection (GIL released in C code) ------
            proj_np = self.scanner.run_fwd_projector(ref_buf_a)

            # --- torch GPU — Beer-Lambert + noise + accumulate ----------------
            curr = torch.as_tensor(proj_np, device=_gt_device)
            del proj_np

            scale = curr_pc * curr_spectrum[e - 10] * system_gain * e
            curr.neg_()
            curr.exp_()
            curr.mul_(scale)

            if add_poisson_noise:
                curr.clamp_(min=0.0)
                curr = torch.poisson(curr)

            if add_system_noise:
                curr.add_(torch.randn_like(curr), alpha=shot_gain)

            projection_buffer_gpu.add_(curr)
            del curr
            pc_sum += scale

            # Wait for prefetch to finish before swapping buffers
            if prefetch_fut is not None:
                prefetch_fut.result()

            # Swap buffers — ref_buf_b (now filled) becomes the active buffer
            ref_buf_a, ref_buf_b = ref_buf_b, ref_buf_a

            kev_iter.set_description(
                f"Processing Energy Level, {e} keV:\t", refresh=True)

        prefetch_pool.shutdown(wait=False)
        del ref_buf_a, ref_buf_b
        del voxel_mat_idx_safe_cpu, bg_mask_cpu, lac_lut_cpu

        # Free ASTRA's cached GPU objects now that the energy sweep is done
        self.scanner.cleanup_astra_cache()

        # Restore large tensors to GPU for downstream stages (decomposer, etc.)
        self.gt_image_3d = self.gt_image_3d.to(_gt_device)
        self.compton_image_3d = self.compton_image_3d.to(_gt_device)
        if hasattr(self, 'pe_image_3d') and torch.is_tensor(self.pe_image_3d):
            self.pe_image_3d = self.pe_image_3d.to(_gt_device)
        if hasattr(self, 'zeff_image_3d') and torch.is_tensor(self.zeff_image_3d):
            self.zeff_image_3d = self.zeff_image_3d.to(_gt_device)

        # Convert accumulated projection to log-attenuation sinogram (on GPU)
        projection_buffer_gpu.clamp_(min=1.0)
        projection_buffer_gpu.log_()
        projection_buffer_gpu.neg_()
        projection_buffer_gpu.add_(log(pc_sum))

        # NOTE: The sinogram is stored in self.scale units (mm⁻¹ when
        # self.scale=0.1).  The correction to cm⁻¹ is applied later in
        # run_reconstructor() — NOT here — because the CDM decomposer
        # expects sinograms in the same units the energy loop produced.

        projection_buffer = projection_buffer_gpu.cpu().numpy()
        del projection_buffer_gpu

        self.logger.info("Sinogram Created ...")

        self._save_output(self.f_loc['sino_file']%spectrum,
                       projection_buffer)

        self.logger.info(f"Time Taken: {time.time() - t0}")

        return projection_buffer
    # --------------------------------------------------------------------------

    def add_scatter_to_ct_projection_slice(self,
                                      add_poisson_noise=True,
                                      add_system_noise=True,
                                      system_gain=5e-3,
                                      spectrum=1,
                                      slice_no=150,
                                      sinogram_buffer=None
                                      ):
        """
        ------------------------------------------------------------------------
        Add deterministic first-order scattering artifacts to the slice. The
        algorithm is adopted from:

        Freud, N., et al. "Deterministic simulation of first-order scattering
        in virtual X-ray imaging." Nuclear Instruments and Methods in Physics
        Research Section B: Beam Interactions with Materials and Atoms 222.1-2
        (2004): 285-300.

        :param add_poisson_noise:   Set to True if Poisson noise is to be added.
        :param add_system_noise:    Set to True if Gaussian shot noise is to be
                                    added.
        :param system_gain:         Gain for Gaussian shot noise
        :param spectrum:            index of the spectrum as specified in self.
                                    xray_source_model.
        :param slice_no:            Slice to which scatter is to be added.
        :param sinogram_buffer:     Optional pre-computed sinogram array.
                                    If provided, avoids re-reading from disk.

        :return
        ------------------------------------------------------------------------
        """

        if sinogram_buffer is not None:
            sino_buf = sinogram_buffer
        elif not os.path.exists(self.f_loc['sino_file']%spectrum):
            sino_buf = self.generate_polychromatic_ct_projection(
                add_poisson_noise=add_poisson_noise,
                add_system_noise=add_system_noise,
                system_gain=system_gain,
                spectrum=spectrum
            )
        else:
            flush_async_io()
            _field = 1 if getattr(self, 'compress_data', False) else 0
            sino_buf = read_fits_data(
                self.f_loc['sino_file'] % spectrum, _field)

        self.sino = sino_buf[slice_no, :, :].copy()

        i = spectrum
        self.logger.info("Generating Polyenergetic Sinograms "
                         "for Spectrum %i ..." % (i))

        curr_spectrum = loadtxt(self.xray_source_model['spectra'][i - 1])[:, 1]
        material_list = unique(list(self.material_curve.keys()))

        # set up the scatter simulation framework
        scatter_sim   = ScatterSimulator(self.scanner,
                                         self.sf_obj_list)
        scatter_sim.set_scatter_calculator(
            self.gt_image_3d[:, :, slice_no].cpu().numpy())

        kev_iter = tqdm(self.keV_range[:curr_spectrum.size])

        # initialize the scattering projection
        scatter_projn = zeros_like(self.sino.shape)

        # Calculate scatter projections for energy level
        for e in kev_iter:
            k = e - self.keV_range[0]
            ref_image = torch.zeros_like(self.compton_image_3d)
            kev_iter.set_description(f"Calculating Scatter at Energy Level, {e} keV:\t",
                                     refresh=True)

            for mat in material_list:
                ref_image = torch.where(
                    self.compton_image_3d == self.mu.material(mat, 'compton'),
                    torch.Tensor([self.material_curve[mat][k]]),
                    ref_image)

            scatter_projn = scatter_sim.get_scatter_projections(
                atten_image=ref_image[:,:,slice_no].cpu().numpy(),
                e=e,
                xray_specs=self.xray_source_model,
                spectrum=curr_spectrum
            )

        pc_sum = self.xray_source_model['dosage'][spectrum-1]

        projn = (self.sino - log(pc_sum))
        projn = np.exp(-projn)

        projn = projn + scatter_projn.T

        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        s_projn = torch.as_tensor(projn, dtype=torch.float, device=_device)

        s_projn = torch.where(s_projn<1,
                              torch.tensor(1.0, device=_device), s_projn)
        s_projn = torch.log(s_projn)
        s_projn = torch.neg(s_projn)
        s_projn = torch.add(s_projn,  log(pc_sum))
        self.scatter_sino = s_projn.cpu().numpy()

        self.scatter_recon = self.scanner.reconstruct_data(self.scatter_sino)

        spectra = [loadtxt(spec)[:self.maxkV,1]
                   for spec in self.xray_source_model['spectra']]

        self.mu.calculate_lac_hu_values('water', spectra)
        mu_w = self.mu.material('water')
        cmin, cmax, offset = -1000, 3.2e4, 0
        scale = self.scanner.recon_params['img_scale']

        self.scatter_recon *= scale
        self.scatter_recon = \
            (self.scatter_recon - mu_w['lac_1']) / mu_w['lac_1'] * 1000 + offset
        self.scatter_recon = clip(self.scatter_recon, cmin, cmax).astype(STORAGE_DTYPE)

        self.recon = self.scanner.reconstruct_data(self.sino)
        self.recon *= scale
        self.recon = (self.recon - mu_w['lac_1']) / mu_w['lac_1'] * 1000 + offset
        self.recon = clip(self.recon, cmin, cmax).astype(STORAGE_DTYPE)

        return self.scatter_sino, self.scatter_recon, self.sino, self.recon

    # --------------------------------------------------------------------------

    def run_bag_generator(self,
                          mode='randomized', 
                          bag_creator_dict=None, 
                          sf_file=None, 
                          sim_args={}):
        """
        ------------------------------------------------------------------------
        Run the Virtual Bag generator block - it can be run either randomized or
        manual mode. In the randomized mode, running the block creates a
        randomized virtual bag with randomly placed objects and randomly assign-
        ed material properties. In the manual mode, running the block reads in
        a shape list (location to a sl_metadata.pyc file) to create the 3D virtual
        bag.

        :param mode:              set the mode to 'manual' or 'randomized'
        :param bag_creator_dict:  dictionary of arguments for BaggageCreator3D
                                  to create a randomized bag
        :param sf_file:           a shape list or path to shape if running in
                                  'manual' mode
        :param sim_args:          additional optional arguments - see
                                  self.create_random_simulation_instance or
                                  self.create_simulation_from_sl_file
        :return: 
        -----------------------------------------------------------------------
        """
        if mode=='randomized':
            self.create_random_simulation_instance(bag_creator_dict, **sim_args)
        
        elif mode=='manual':
            self.create_simulation_from_sl_file(sf_file, **sim_args)
    # --------------------------------------------------------------------------

    def _to_arcname(self, abs_path):
        """Convert an absolute path to a forward-slash archive-relative path."""
        abs_norm = os.path.normpath(abs_path).replace(os.sep, '/')
        sim_norm = os.path.normpath(
            self.f_loc['simulation_dir']).replace(os.sep, '/').rstrip('/')
        if abs_norm.startswith(sim_norm + '/'):
            return abs_norm[len(sim_norm) + 1:]
        return abs_norm.split('/')[-1]

    def _save_output(self, path, array, compress=None):
        """Save an array — routes to archive (monolithic) or async FITS.

        :param path:      absolute path or relative to simulation_dir
        :param array:     numpy array to save
        :param compress:  override compress flag; defaults to self.compress_data
        """
        if compress is None:
            compress = self.compress_data
        if self.monolithic_output and self.archive is not None:
            arcname = self._to_arcname(path)
            self.archive.add_fits(arcname, array, compress)
            self.logger.info(f"Archived: {arcname}")
        else:
            save_fits_data_async(path, array, compress)
            self.logger.info(f"Saving: {path}")

    def _save_npz(self, path, *args, **arrays):
        """Save numpy arrays as .npz — routes to archive or disk."""
        if self.monolithic_output and self.archive is not None:
            kw = dict(arrays)
            for i, a in enumerate(args):
                kw[f'arr_{i}'] = a
            arcname = self._to_arcname(path)
            self.archive.add_npz(arcname, **kw)
            self.logger.info(f"Archived: {arcname}")
        else:
            savez_compressed(path, *args, **arrays)
            self.logger.info(f"Saving: {path}")

    def _save_pickle(self, path, obj):
        """Save a pickle — routes to archive or disk."""
        if self.monolithic_output and self.archive is not None:
            self.archive.add_pickle(self._to_arcname(path), obj)
        else:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
    # --------------------------------------------------------------------------

    def run_fwd_model(self,
                      add_poisson_noise=True,
                      add_system_noise=True,
                      system_gain=5e-4):
        """
        ------------------------------------------------------------------------
        Run the forward X-ray modelling block - generates polychromatic proj-
        ections for the entire system by iterating the function
        self.generate_polychromatic_ct_projection() over all the spectra in the
        model.

        :param add_poisson_noise:   Set to True if Poisson noise is to be added.
        :param add_system_noise:    Set to True if Gaussian shot noise is to be
                                    added.
        :param system_gain:         Gain for Gaussian shot noise
        :return:
        ------------------------------------------------------------------------
        """

        sinograms = []
        for spec_no in range(self.xray_source_model['num_spectra']):
            sino = self.generate_polychromatic_ct_projection(
                add_poisson_noise=add_poisson_noise,
                add_system_noise=add_system_noise,
                system_gain=system_gain,
                spectrum=spec_no + 1
            )
            sinograms.append(sino)
            torch.cuda.empty_cache()

        self.logger.info("Xray data generated")
        self.logger.info("=" * 80)

        # Stash raw (projector-format) sinograms so callers like
        # run_fwd_model_with_motion_artifacts() can interleave from
        # memory instead of re-reading from disk.
        self._raw_sinograms = sinograms

        # Use in-memory sinograms directly — no disk read-back needed.
        # The async FITS writes are still running in background for persistence.
        if self.xray_source_model['num_spectra']==2:
            data1 = moveaxis(sinograms[0], -1, 0)
            data1 = data1[:, :, ::-1]
            data2 = moveaxis(sinograms[1], -1, 0)
            data2 = data2[:, :, ::-1]

            self.data1 = np.ascontiguousarray(data1)
            self.data2 = np.ascontiguousarray(data2)

        elif self.xray_source_model['num_spectra']==1:
            data = moveaxis(sinograms[0], -1, 0)
            data = data[:, :, ::-1]

            self.data = np.ascontiguousarray(data)
    # --------------------------------------------------------------------------

    def run_fwd_model_with_motion_artifacts(self,
                                            n_steps=3,
                                            blur_res=6,
                                            mode='bag',
                                            bag_params=None,
                                            lqd_params=None,
                                            obj_params=None,
                                            fwd_model_args=None
                                            ):
        """
        -----------------------------------------------------------------------

        :param n_steps:
        :param blur_res:
        :param mode:
        :param bag_params:
        :param lqd_params:
        :param obj_params:
        :return:
        -----------------------------------------------------------------------
        """

        if fwd_model_args is None:
            fwd_model_args = dict(add_poisson_noise=True,
                                  add_system_noise=True,
                                  system_gain=5e-4)

        assert mode in ['bag', 'objects']

        # Collect raw (projector-format) sinograms per sequence so the
        # interleaving step can work from memory instead of re-reading FITS.
        seq_sinograms = []

        self.logger.info("Creating Data for Original Virtual Bag ...")
        self.f_loc['sino_file'] = os.path.join(self.f_loc['sino_dir'],
                                               'sino_%i_seq_00.fits.gz')
        self.run_fwd_model(**fwd_model_args)
        seq_sinograms.append(self._raw_sinograms)

        if mode=='bag':

            if bag_params is None:
                bag_params = dict(n_seqs=4,
                                  rotate=True,
                                  x_tol=4,
                                  t_tol=3,
                                  fixed_x=True)

            n_seqs = bag_params['objects']

            for s in range(1, bag_params['n_seqs']):

                self.logger.info("="*80)
                self.logger.info(f"Simulating Sequence {s}")
                # the motion translation, rotation parameters
                x_tol = np.random.choice(range(-bag_params['x_tol'],
                                                bag_params['x_tol']),
                                         size=3
                                         )
                t_tol = np.random.choice(range(-bag_params['t_tol'],
                                                bag_params['t_tol']),
                                         size=3)

                # if no vertical movement is allowed
                if bag_params['fixed_x']:
                    x_tol[0] = 0
                    t_tol[0], t_tol[2] = 0, 0

                self.gt_image_3d = self.gt_image_3d.cpu().numpy()
                self.gt_image_3d = sptx.rotate(self.gt_image_3d, t_tol[0],
                                               axes=(0,1)).astype(NP_INT)
                self.gt_image_3d = sptx.rotate(self.gt_image_3d, t_tol[1],
                                               axes=(1,2)).astype(NP_INT)
                self.gt_image_3d = sptx.rotate(self.gt_image_3d, t_tol[2],
                                               axes=(2, 0)).astype(NP_INT)

                self.gt_image_3d = sptx.shift(self.gt_image_3d,
                                              x_tol).astype(NP_INT)
                self.gt_image_3d = torch.from_numpy(self.gt_image_3d).to('cuda')

                self.compton_image_3d = torch.zeros_like(self.gt_image_3d,
                                                         dtype=T_FLOAT)

                # create compton image for shifted virtual bag
                for sf_obj in self.sf_obj_list:
                    self.compton_image_3d = torch.where(
                        self.gt_image_3d == sf_obj['label'],
                        torch.Tensor([self.mu.material(sf_obj['material'],
                                                       'compton')]),
                        self.compton_image_3d)

                    if sf_obj['lqd_flag']:
                        self.compton_image_3d = torch.where(
                            self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                            torch.Tensor([
                                self.mu.material(sf_obj['lqd_param']['lqd_material'],
                                'compton')]),
                            self.compton_image_3d)

                self.f_loc['sino_file'] =  os.path.join(self.f_loc['sino_dir'],
                                                        f"sino_%i_seq_{s:02d}.fits.gz")
                self.run_fwd_model(**fwd_model_args)
                seq_sinograms.append(self._raw_sinograms)
                self.logger.info("="*80)
                self.logger.info("="*80)

                del self.compton_image_3d
                torch.cuda.empty_cache()

        elif mode=='objects':

            if obj_params is None:
                obj_params = dict(n_seqs=6,
                                  rotate=True,
                                  x_tol=6,
                                  t_tol=5,
                                  objects=10,
                                  fixed_x=True)

            obj_labels = [x['label'] for x in self.sf_obj_list
                          if x['label'] not in [1,2,3]]

            if isinstance(obj_params['objects'], int):
                obj_params['objects'] = np.random.choice(obj_labels,
                                                         size=obj_params['objects'])
            elif isinstance(obj_params['objects'], list):
                pass
            else:
                raise TypeError("Datatype for obj_params['objects'] not recognized!")

            self.gt_image_3d = self.gt_image_3d.cpu().numpy()
            orig_gt_image = self.gt_image_3d.copy()

            n_seqs = obj_params['n_seqs']

            for s in range(1, obj_params['n_seqs']):

                self.logger.info("="*80)
                self.logger.info(f"Simulating Sequence {s}")
                # the motion translation, rotation parameters
                x_tol = np.random.choice(range(-obj_params['x_tol'],
                                                obj_params['x_tol']),
                                         size=3)
                t_tol = np.random.choice(range(-obj_params['t_tol'],
                                                obj_params['t_tol']),
                                         size=3)

                # if no vertical movement is allowed
                if obj_params['fixed_x']:
                    x_tol[0] = 0
                    t_tol[0], t_tol[2] = 0, 0

                masked_gt_image  = orig_gt_image.copy()
                moving_obj_vol = zeros_like(masked_gt_image)

                for i in obj_params['objects']:
                    moving_obj_vol[orig_gt_image==i] = i
                    masked_gt_image[orig_gt_image==i] = 0

                # nz = moving_obj_vol.nonzero()

                moving_obj_vol = sptx.rotate(moving_obj_vol,
                                             t_tol[0],
                                             axes=(0,1),
                                             reshape=False).astype(NP_INT)
                moving_obj_vol = sptx.rotate(moving_obj_vol,
                                             t_tol[1],
                                             axes=(1,2),
                                             reshape=False).astype(NP_INT)
                moving_obj_vol = sptx.rotate(moving_obj_vol,
                                             t_tol[2],
                                             axes=(2, 0),
                                             reshape=False).astype(NP_INT)

                moving_obj_vol = sptx.shift(moving_obj_vol,
                                            x_tol).astype(NP_INT)

                masked_gt_image[moving_obj_vol>0] = moving_obj_vol[moving_obj_vol>0]
                self.gt_image_3d = torch.from_numpy(masked_gt_image).to('cuda')

                self.compton_image_3d = torch.zeros_like(self.gt_image_3d,
                                                         dtype=T_FLOAT)

                # create compton image for shifted virtual bag
                for sf_obj in self.sf_obj_list:
                    self.compton_image_3d = torch.where(
                        self.gt_image_3d == sf_obj['label'],
                        torch.Tensor([self.mu.material(sf_obj['material'],
                                                       'compton')]),
                        self.compton_image_3d)

                    if sf_obj['lqd_flag']:
                        self.compton_image_3d = torch.where(
                            self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                            torch.Tensor([
                                self.mu.material(sf_obj['lqd_param']['lqd_material'],
                                'compton')]),
                            self.compton_image_3d)

                self.f_loc['sino_file'] =  os.path.join(self.f_loc['sino_dir'],
                                                        f"sino_%i_seq_{s:02d}.fits.gz")
                self.run_fwd_model(**fwd_model_args)
                seq_sinograms.append(self._raw_sinograms)
                self.logger.info("="*80)
                self.logger.info("="*80)

                del self.compton_image_3d
                torch.cuda.empty_cache()

        # Interleave motion-sequence sinograms from in-memory buffers.
        # Each call to run_fwd_model() stashes raw sinograms in
        # self._raw_sinograms — collect them per sequence to avoid
        # re-reading from disk.
        if self.xray_source_model['num_spectra']==1:

            self.data = zeros((self.scanner.machine_geometry['det_row_count'],
                               self.scanner.recon_geometry['n_views'],
                               self.scanner.machine_geometry['det_col_count']))

            for s, raw_sinos in enumerate(seq_sinograms):
                self.data[:, s::n_seqs, :] = raw_sinos[0][:, s::n_seqs, :]

            self.f_loc['sino_file'] = os.path.join(self.f_loc['sino_dir'],
                                                    'sino_%i.fits.gz')
            self._save_output(self.f_loc['sino_file'] % 1,
                           self.data)

            self.data = moveaxis(self.data, -1, 0)
            self.data = self.data[:, :, ::-1]

        elif self.xray_source_model['num_spectra']==2:

            self.data1 = zeros((self.scanner.machine_geometry['det_row_count'],
                                self.scanner.machine_geometry['n_views'],
                                self.scanner.machine_geometry['det_col_count']))
            self.data2 = zeros((self.scanner.machine_geometry['det_row_count'],
                                self.scanner.machine_geometry['n_views'],
                                self.scanner.machine_geometry['det_col_count']))

            for s, raw_sinos in enumerate(seq_sinograms):
                self.data1[:, s::n_seqs, :] = raw_sinos[0][:, s::n_seqs, :]
                self.data2[:, s::n_seqs, :] = raw_sinos[1][:, s::n_seqs, :]

            self.f_loc['sino_file'] = 'sino_%i.fits.gz'
            self._save_output(self.f_loc['sino_file'] % 1, self.data1)

            self.data1 = moveaxis(self.data1, -1, 0)
            self.data1 = self.data1[:, :, ::-1]

            self._save_output(self.f_loc['sino_file'] % 2, self.data2)

            self.data2 = moveaxis(self.data2, -1, 0)
            self.data2 = self.data2[:, :, ::-1]
    # -------------------------------------------------------------------------

    def run_decomposer(self,
                       type='cdm',
                       decomposer_args=None,
                       basis_fn=None,
                       save_sino=False):

        """
        ------------------------------------------------------------------------
        Run the Dual Energy Decomposition block -  uses either the CDM or SIRZ
        or LUTD to process Dual Energy data

        :param type:            select from {'cdm' | 'sirz' | 'lutd' }
        :param decomposer_args: additional arguments for DE decomposition
        :return:
        ------------------------------------------------------------------------
        """
        self.DECOMPOSER_FLAG = True
        self.decomposer_type = type
        self.basis_fn = basis_fn
        self.save_de_sino = save_sino

        assert type=='cdm', "Open-source version only support CDM method " \
                            "for dual energy decomposition"

        if type in ['cdm', 'sirz']:

            # Initialize CDM reconstructor
            sim_specs = dict(
                spctr_h_fname=self.xray_source_model['spectra'][0],
                spctr_l_fname=self.xray_source_model['spectra'][1],
                photon_count_high=self.xray_source_model['dosage'][0],
                photon_count_low=self.xray_source_model['dosage'][1],
                nangs=self.data1.shape[2],
                nbins=self.data1.shape[0],
                projector='cpu'
            )

            cdm_sim = CDMDecomposer(**sim_specs)
            # -----------------------------------------------------------------------------

            # Set basis function if not using Compton-PE basis
            if basis_fn is not None:
                cdm_sim.set_basis_functions(**basis_fn)

            nrows = self.data1.shape[1]

            # The decomposer's basis functions (Klein-Nishina, PE=e^-3)
            # are defined for LAC in cm⁻¹.  Convert sinograms from
            # self.scale units (mm⁻¹) to cm⁻¹ before decomposition.
            decomp_scale = 1.0 / self.scale if abs(self.scale - 1.0) > 1e-6 else 1.0

            solver = decomposer_args['cdm_solver']
            cdm_type = decomposer_args['cdm_type']

            if solver == 'gpu' and nrows > 1:
                # ---- Batched GPU path: all rows in a single GpuFit call ------
                # GpuFit handles millions of independent fits efficiently.
                # Stacking all rows into one call avoids nrows separate GPU
                # kernel launches + data transfers.
                self.logger.info(
                    f"Decomposing {nrows} rows in a single batched GPU call")

                # Stack all rows: shape (nbins, nrows, nangs) → flatten
                # across rows so GpuFit sees nbins*nrows*nangs pixel pairs.
                data1_scaled = self.data1 * decomp_scale
                data2_scaled = self.data2 * decomp_scale

                # Reshape to (nbins*nrows, nangs) per 2D slice convention,
                # then let decompose_dect_sinograms flatten internally.
                nbins, _, nangs = self.data1.shape

                # Temporarily override sino_shape to match the full 3D volume
                orig_shape = cdm_sim.sino_shape
                orig_npxls = cdm_sim.n_sino_pxls
                cdm_sim.sino_shape = (nbins * nrows, nangs)
                cdm_sim.n_sino_pxls = nbins * nrows * nangs

                cdm_sim.init_val = decomposer_args['init_val']
                sino_pe_flat, sino_c_flat = \
                    cdm_sim.decompose_dect_sinograms(
                        data1_scaled.reshape(nbins * nrows, nangs),
                        data2_scaled.reshape(nbins * nrows, nangs),
                        solver='gpu',
                        type=cdm_type
                    )

                # Restore and reshape back to 3D
                cdm_sim.sino_shape = orig_shape
                cdm_sim.n_sino_pxls = orig_npxls
                sino_pe = sino_pe_flat.reshape(nbins, nrows, nangs)
                sino_c = sino_c_flat.reshape(nbins, nrows, nangs)

                del data1_scaled, data2_scaled
                torch.cuda.empty_cache()
            else:
                # ---- Row-by-row fallback (CPU / vec solver) ------------------
                sino_pe = zeros_like(self.data1)
                sino_c = zeros_like(self.data2)

                for i in range(nrows):
                    self.logger.info("Row %d:" % i)
                    cdm_sim.init_val = decomposer_args['init_val']
                    sino_pe[:, i, :], sino_c[:, i, :] = \
                        cdm_sim.decompose_dect_sinograms(
                            self.data1[:, i, :] * decomp_scale,
                            self.data2[:, i, :] * decomp_scale,
                            solver=solver,
                            type=cdm_type
                        )

            self.sino_c = sino_c.copy()
            self.sino_pe = sino_pe.copy()

    # --------------------------------------------------------------------------

    def run_reconstructor(self,
                          img_type='HU',
                          recon='fbp',
                          plot_stats=True,
                          fname=None):
        """
        ------------------------------------------------------------------------
        Run the reconstructor block - uses the methods from ScannerTemplate to
        run the reconstructor for the scanner geometry that was used to
        initialize the ScannerTemplate class.

        :param img_type:    select the image unit: {'HU' | 'MHU' | 'LAC'}
        :param recon:       select reconstruction algo: {'sirt' | 'fbp'}
        :return:
        ------------------------------------------------------------------------
        """

        # reconstruct images from the sinograms
        if recon != 'fbp':
            self.scanner.update_recon_algo(recon)

        # The energy loop stores sinograms in self.scale units (mm⁻¹
        # when self.scale=0.1) for noise model compatibility.
        # Convert to cm⁻¹ before reconstruction so that FBP × img_scale
        # produces correct LAC for HU conversion.
        sino_correction = 1.0 / self.scale if abs(self.scale - 1.0) > 1e-6 else 1.0

        # ---- Beam Hardening Correction (sinogram domain) --------------------
        bhc_correctors = []
        if getattr(self, 'apply_bhc', True):
            try:
                from lib.forward_model.bhc import BeamHardeningCorrector
                for spec_path in self.xray_source_model['spectra']:
                    bhc = BeamHardeningCorrector.from_debisim(
                        self.mu, spec_path, max_kev=self.maxkV)
                    bhc_correctors.append(bhc)
                    self.logger.info(
                        f"BHC LUT built: E_eff={bhc.e_eff:.1f} keV, "
                        f"mu_mono={bhc.mu_mono:.4f} cm^-1")
            except Exception as e:
                self.logger.warning(f"BHC initialization failed: {e}")
                bhc_correctors = []

        def _apply_bhc(sino, spectrum_idx):
            """Apply BHC to sinogram if corrector is available."""
            sino_cm = sino * sino_correction
            if spectrum_idx < len(bhc_correctors):
                return bhc_correctors[spectrum_idx].correct(sino_cm)
            return sino_cm

        if self.xray_source_model['num_spectra']==2:
            sino1_corrected = _apply_bhc(self.data1, 0)
            image_1 = self.scanner.reconstruct_data(
                sino1_corrected,
                full_range=True, append_air_turns=True)
            del sino1_corrected

            self.logger.info("Reconstructed LAC Image 1 ...")

            sino2_corrected = _apply_bhc(self.data2, 1)
            image_2 = self.scanner.reconstruct_data(
                sino2_corrected,
                full_range=True, append_air_turns=True)
            del sino2_corrected
            self.logger.info("Reconstructed LAC Image 2 ...")

            del self.data1, self.data2
        elif self.xray_source_model['num_spectra']==1:
            sino_corrected = _apply_bhc(self.data, 0)
            image_1 = self.scanner.reconstruct_data(
                sino_corrected,
                full_range=True, append_air_turns=True)
            del sino_corrected
            self.logger.info("Reconstructed LAC Image ...")
            del self.data

        # reconstruct any decomposed line integrals

        if self.DECOMPOSER_FLAG:

            image_c = self.scanner.reconstruct_data(self.sino_c, full_range=True,
                                               append_air_turns=True)
            self.logger.info("Reconstructed Compton Image ...")

            image_pe = self.scanner.reconstruct_data(self.sino_pe, full_range=True,
                                                append_air_turns=True)
            self.logger.info("Reconstructed PE Image ...")

            if self.decomposer_type in ['cdm', 'lutd']:
                image_z = effective_atomic_number(image_pe, image_c)
                img_suffixes = ['c', 'pe']
                self.logger.info("Created Zeff Image ...")

            elif self.decomposer_type=='sirz':
                image_z, image_rho =  self.sirz_decomp.run_sirz2_decomp(self.sino_c,
                                                                        self.sino_pe)
                img_suffixes = ['b1', 'b2']
                self.logger.info("Created Ze-Rhoe Images ...")

            if self.save_de_sino:
                if fname is None:
                    out_fname = os.path.join(self.f_loc['sino_dir'],
                                             'sino_%s.npz' % (img_suffixes[0]))
                else:
                    out_fname = os.path.join(self.f_loc['sino_dir'],
                                             fname % (img_suffixes[0]))

                self.sino_c = moveaxis(self.sino_c, -1, 0)
                self._save_npz(out_fname, self.sino_c)
            del self.sino_c

            if self.save_de_sino:
                if fname is None:
                    out_fname = os.path.join(self.f_loc['sino_dir'],
                                             'sino_%s.npz' % (img_suffixes[1]))
                else:
                    out_fname = os.path.join(self.f_loc['sino_dir'],
                                             fname % (img_suffixes[1]))

                self.sino_pe = moveaxis(self.sino_pe, -1, 0)
                self._save_npz(out_fname, self.sino_pe)
            del self.sino_pe

        scale = self.scanner.recon_params['img_scale']

        spectra = [loadtxt(spec)[:self.maxkV,1]
                   for spec in self.xray_source_model['spectra']]

        self.mu.calculate_lac_hu_values('water', spectra)
        mu_w = self.scanner.recon_params['mu_w'] if self.scanner.recon_params['mu_w'] \
                                                 is not None else self.mu.material('water')

        # convert to Hounsfield or Modified Hounsfield units
        if img_type=='HU':  cmin, cmax, offset = -1000, 3.2e4, 0
        if img_type=='MHU': cmin, cmax, offset = 0, 3.2e4, 1000

        if img_type in ['HU', 'MHU']:

            if self.xray_source_model['num_spectra'] == 2:

                if self.scanner.machine_geometry['scanner_name'] in (
                        'default_parallelbeam', 'parallel_custom'):
                    # FBP returns (n_slices, im_x, im_y) — reorder to
                    # (im_x, im_y, n_slices) to match GT volume layout.
                    # No flip — axis orientation must match GT labels for
                    # correct HU statistics and DICOM ROI alignment.
                    image_1 = np.ascontiguousarray(
                        moveaxis(image_1, 0, -1))
                    image_2 = np.ascontiguousarray(
                        moveaxis(image_2, 0, -1))

                image_1 *= scale
                image_1 = (image_1 - mu_w['lac_1']) / mu_w['lac_1'] * 1000 + offset
                image_1 = clip(image_1, cmin, cmax).astype(STORAGE_DTYPE)

                out_fname = os.path.join(self.f_loc['image_dir'],
                                         self.f_loc['img_file'] % 1)


                self._save_output(out_fname, image_1)

                image_2 *= scale
                image_2 = (image_2 - mu_w['lac_2']) / mu_w['lac_2'] * 1000 + offset
                image_2  = clip(image_2,  cmin, cmax).astype(STORAGE_DTYPE)

                # Save reconstructed images
                out_fname = os.path.join(self.f_loc['image_dir'],
                                         self.f_loc['img_file'] % 2)

                self._save_output(out_fname, image_2)

                # Cache in memory so save_dicom_output() doesn't re-read from disk
                self._recon_images_cache = [image_1, image_2]

            elif self.xray_source_model['num_spectra'] == 1:
                image_1 *= scale
                image_1 = (image_1 - mu_w['lac_1']) / mu_w['lac_1'] * 1000 + offset
                image_1 = clip(image_1, cmin, cmax).astype(STORAGE_DTYPE)

                if self.scanner.machine_geometry['scanner_name'] in (
                        'default_parallelbeam', 'parallel_custom'):
                    image_1 = np.ascontiguousarray(
                        moveaxis(image_1, 0, -1))

                out_fname = os.path.join(self.f_loc['image_dir'],
                                         self.f_loc['img_file']%1)

                self._save_output(out_fname, image_1)

                # Cache in memory so save_dicom_output() doesn't re-read from disk
                self._recon_images_cache = [image_1]

            if self.DECOMPOSER_FLAG:
                if self.scanner.machine_geometry['scanner_name'] in (
                        'default_parallelbeam', 'parallel_custom'):
                    image_c = np.ascontiguousarray(
                        moveaxis(image_c, 0, -1))
                    image_pe = np.ascontiguousarray(
                        moveaxis(image_pe, 0, -1))
                    image_z = np.ascontiguousarray(
                        moveaxis(image_z, 0, -1))

                # Compton and PE images are basis decomposition outputs —
                # they are NOT in LAC units, so the HU formula does not
                # apply.  Scale by img_scale (same as LAC images) to undo
                # the self.scale used in the energy loop, then store as-is.
                #
                # Z_eff is a dimensionless quantity (effective atomic number)
                # and should never be HU-normalized.
                image_c *= scale
                image_pe *= scale
                # Mask Z_eff to 0 in air regions (where LAC is near -1000 HU)
                # so the viewer doesn't display noise as atomic number.
                air_threshold = cmin + 200  # -800 HU
                if image_z.shape != image_1.shape:
                    common = tuple(builtins.min(a, b) for a, b in
                                   zip(image_z.shape, image_1.shape))
                    image_z = image_z[:common[0], :common[1], :common[2]]
                    image_1_crop = image_1[:common[0], :common[1], :common[2]]
                    image_2_crop = image_2[:common[0], :common[1], :common[2]]
                    image_z[image_1_crop < air_threshold] = 0
                    image_z[image_2_crop < air_threshold] = 0
                else:
                    image_z[image_1 < air_threshold] = 0
                    image_z[image_2 < air_threshold] = 0

                # Basis decomposition outputs have fractional values —
                # store as float32, not int16 (which truncates 0.54 → 0).
                # Negative values from unconstrained LM are preserved
                # (realistic — real scanners produce them from noise).
                # Z_eff is clamped to [0, 92] since negative Z is undefined.
                image_z = clip(image_z, 0, 92).astype(np.float32)
                image_c = image_c.astype(np.float32)
                image_pe = image_pe.astype(np.float32)

                out_fname = os.path.join(self.f_loc['image_dir'],
                                         self.f_loc['img_file'].replace('%i', '%s') % img_suffixes[0])

                self._save_output(out_fname, image_c)

                out_fname = os.path.join(self.f_loc['image_dir'],
                                         self.f_loc['img_file'].replace('%i', '%s') % img_suffixes[1])

                self._save_output(out_fname, image_pe)

                out_fname = os.path.join(self.f_loc['image_dir'],
                                         self.f_loc['img_file'].replace('%i', '%s') % 'z')

                self._save_output(out_fname, image_z)

                del image_c, image_pe, image_z, image_1, image_2

                if self.decomposer_type=='sirz':

                    image_rho *= scale
                    image_rho = (image_rho - mu_w['density']) / mu_w['density'] * 1000 + offset
                    image_rho = clip(image_rho, cmin, cmax).astype(STORAGE_DTYPE)

                    out_fname = os.path.join(self.f_loc['image_dir'],
                                             'recon_image_%s.fits.gz' % 'rho')
    
                    self._save_output(out_fname, image_rho)

            # Wait for all async FITS writes to complete before exiting
            flush_async_io()
            self.logger.info("All FITS files saved.")
    # --------------------------------------------------------------------------

    def save_dicom_output(self):
        """
        Save reconstructed images as DICOM CT series and create an RT-Struct
        with threat ROI contours. Each series folder is named with its kV.
        """
        from lib.misc.util import save_dicom_output as _save_dicom

        # Extract per-spectrum kV from spectra filenames or kVp field
        num_spectra = self.xray_source_model['num_spectra']
        spectra_paths = self.xray_source_model.get('spectra', [])
        kvp = self.xray_source_model.get('kVp', None)

        kv_labels = []
        for s in range(num_spectra):
            if s < len(spectra_paths):
                # Try to extract kV from filename like 'example_spectrum_130kV.txt'
                import re
                match = re.search(r'(\d+)\s*[kK][vV]', os.path.basename(spectra_paths[s]))
                if match:
                    kv_labels.append(f'{match.group(1)}kV')
                    continue
            # Fallback: use kVp field or index
            if kvp is not None:
                kv_labels.append(f'{kvp}kV')
            else:
                kv_labels.append(f'spectrum_{s+1}')

        # Use cached reconstructed images if available (avoids FITS re-read)
        recon_images = {}
        cached = getattr(self, '_recon_images_cache', None)
        if cached is not None:
            for s in range(builtins.min(num_spectra, len(cached))):
                recon_images[kv_labels[s]] = cached[s]
            del self._recon_images_cache
        else:
            # Fallback: re-read from FITS files
            for s in range(1, num_spectra + 1):
                fpath = os.path.join(self.f_loc['image_dir'],
                                     self.f_loc['img_file'] % s)
                if os.path.exists(fpath):
                    recon_images[kv_labels[s - 1]] = read_fits_data(fpath, 0)

        gt_label = self.gt_image_3d
        if hasattr(gt_label, 'cpu'):
            gt_label = gt_label.cpu().numpy()
        gt_label = gt_label.astype(STORAGE_DTYPE)

        # Build scan metadata from scanner and source model
        mg = self.scanner_geometry
        rp = self.scanner.recon_params
        dosage_list = self.xray_source_model.get('dosage', [])
        spectra_paths = self.xray_source_model.get('spectra', [])
        scan_metadata = dict(
            # Scanner identification
            scanner_name=mg.get('scanner_name', 'DEBISim2'),
            geometry=self.scanner.geom,
            scan_type=self.scanner.scan,
            # Geometry
            source_origin=mg.get('source_origin', ''),
            origin_det=mg.get('origin_det', ''),
            gantry_diameter=mg.get('gantry_diameter', ''),
            fov=mg.get('gantry_diameter', ''),
            # Detector
            det_row_count=mg.get('det_row_count', ''),
            det_col_count=mg.get('det_col_count', ''),
            det_spacing_x=mg.get('det_spacing_x', ''),
            det_spacing_y=mg.get('det_spacing_y', ''),
            sens_spacing_x=mg.get('sens_spacing_x', ''),
            sens_spacing_y=mg.get('sens_spacing_y', ''),
            anode_angle=mg.get('anode_angle', ''),
            # Acquisition
            num_views=rp.get('n_views', ''),
            view_range=rp.get('view_range', ''),
            num_spectra=self.xray_source_model.get('num_spectra', 1),
            kVp=kvp if kvp is not None else '',
            dosage=dosage_list[0] if len(dosage_list) > 0 else '',
            dosage_list=dosage_list,
            spectra_files=[os.path.basename(s) for s in spectra_paths],
            # Reconstruction
            recon_algo=self.scanner.recon,
            image_dims=rp.get('image_dims', []),
            img_scale=rp.get('img_scale', ''),
        )

        # Derive pixel spacing from scanner geometry
        pixel_sz = mg.get('det_spacing_y', 1.0)
        slice_sz = mg.get('det_spacing_x', pixel_sz)

        archive = self.archive if self.monolithic_output else None
        # When archiving, pass a relative prefix so all DICOM arcnames
        # are relative paths with forward slashes (e.g. images/dicom/...).
        dicom_image_dir = (self._to_arcname(self.f_loc['image_dir'])
                           if archive else self.f_loc['image_dir'])
        _save_dicom(
            image_dir=dicom_image_dir,
            recon_images=recon_images,
            gt_label_volume=gt_label,
            sf_obj_list=self.sf_obj_list,
            pixel_spacing=(float(pixel_sz), float(pixel_sz)),
            slice_thickness=float(slice_sz),
            scan_metadata=scan_metadata,
            mu_handler=self.mu,
            archive=archive,
        )

        if archive is not None:
            self.logger.info("Archived DICOM output")
        else:
            dicom_dir = os.path.join(self.f_loc['image_dir'], 'dicom')
            self.logger.info("DICOM output saved to %s" % dicom_dir)
    # --------------------------------------------------------------------------

    def save_dect_ground_truth_images(self, images=['gt']):
        """
        ------------------------------------------------------------------------
        Save 3D image files generated by the phantom voxelizer as .fits.gz files.

        :param images:  The ground truth images to be saved. Options = {'gt',
                        'compton', 'pe', 'zeff', 'lac_1', 'lac_2'}
        :return
        ------------------------------------------------------------------------
        """

        if 'gt' in images:
            self._save_output(self.f_loc['gt_image'],
                           self.gt_image_3d.cpu().numpy().astype(STORAGE_DTYPE))



        if 'compton' in images:
            compton_image_file = os.path.join(self.f_loc['gt_dir'],
                                              'gt_compton_image.fits.gz')

            self._save_output(compton_image_file,
                           self.compton_image_3d.cpu().numpy())



        if 'pe' in images:
            pe_image_file = os.path.join(self.f_loc['gt_dir'],
                                         'gt_pe_image.fits.gz')

            pe_image_3d = torch.zeros_like(self.gt_image_3d).to(T_FLOAT)

            for sf_obj in self.sf_obj_list:
                pe_image_3d = torch.where(self.gt_image_3d == sf_obj['label'],
                                          torch.Tensor([self.mu.material(sf_obj['material'],
                                                                         'pe')]),
                                          pe_image_3d)

                if sf_obj['lqd_flag']:
                    lac_1_image_3d = torch.where(self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                                                 torch.Tensor([self.mu.material(sf_obj['lqd_param'][
                                                                                    'lqd_material'],
                                                                                'pe')]),
                                                 pe_image_3d)

            self._save_output(pe_image_file, pe_image_3d.cpu().numpy())
            del pe_image_3d


        if 'zeff' in images:
            zeff_image_file = os.path.join(self.f_loc['gt_dir'],
                                           'gt_zeff_image.fits.gz')

            zeff_image_3d = torch.zeros_like(self.gt_image_3d).to(T_FLOAT)

            for sf_obj in self.sf_obj_list:
                zeff_image_3d = torch.where(self.gt_image_3d == sf_obj['label'],
                                            torch.Tensor([self.mu.material(sf_obj['material'],
                                                                           'z')]),
                                            zeff_image_3d)

                if sf_obj['lqd_flag']:
                    lac_1_image_3d = torch.where(self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                                                 torch.Tensor([self.mu.material(sf_obj['lqd_param'][
                                                                                    'lqd_material'],
                                                                                'z')]),
                                                 zeff_image_3d)

            self._save_output(zeff_image_file, zeff_image_3d.cpu().numpy())
            del zeff_image_3d


        if 'lac' in images:
            lac_image_file = os.path.join(self.f_loc['gt_dir'],
                                            'gt_lac_image.fits.gz')

            lac_image_3d = torch.zeros_like(self.gt_image_3d).to(T_FLOAT)

            for sf_obj in self.sf_obj_list:
                lac_image_3d = torch.where(self.gt_image_3d == sf_obj['label'],
                                             torch.Tensor([self.mu.material(sf_obj['material'],
                                                                            'lac')]),
                                             lac_image_3d)

                if sf_obj['lqd_flag']:
                    lac_1_image_3d = torch.where(self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                                                 torch.Tensor([self.mu.material(sf_obj['lqd_param']['lqd_material'],
                                                                                'lac')]),
                                                 lac_1_image_3d)

            self._save_output(lac_image_file, lac_image_3d.cpu().numpy())
            del lac_image_3d


        if 'lac_1' in images:
            lac_1_image_file = os.path.join(self.f_loc['gt_dir'],
                                            'gt_lac_1_image.fits.gz')

            lac_1_image_3d = torch.zeros_like(self.gt_image_3d).to(T_FLOAT)

            for sf_obj in self.sf_obj_list:
                lac_1_image_3d = torch.where(self.gt_image_3d == sf_obj['label'],
                                             torch.Tensor([self.mu.material(sf_obj['material'],
                                                                            'lac_1')]),
                                             lac_1_image_3d)

                if sf_obj['lqd_flag']:
                    lac_1_image_3d = torch.where(self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                                                 torch.Tensor([self.mu.material(sf_obj['lqd_param'][
                                                                                    'lqd_material'],
                                                                                'lac_1')]),
                                                 lac_1_image_3d)

            self._save_output(lac_1_image_file, lac_1_image_3d.cpu().numpy())
            del lac_1_image_3d


        if 'lac_2' in images:
            lac_2_image_file = os.path.join(self.f_loc['gt_dir'],
                                            'gt_lac_2_image.fits.gz')

            lac_2_image_3d = torch.zeros_like(self.gt_image_3d).to(T_FLOAT)

            for sf_obj in self.sf_obj_list:
                lac_2_image_3d = torch.where(self.gt_image_3d == sf_obj['label'],
                                             torch.Tensor([self.mu.material(sf_obj['material'],
                                                                            'lac_2')]),
                                             lac_2_image_3d)

                if sf_obj['lqd_flag']:
                    lac_1_image_3d = torch.where(self.gt_image_3d == sf_obj['lqd_param']['lqd_label'],
                                                 torch.Tensor([self.mu.material(sf_obj['lqd_param'][
                                                                                    'lqd_material'],
                                                                                'lac_2')]),
                                                 lac_2_image_3d)

            self._save_output(lac_2_image_file, lac_2_image_3d.cpu().numpy())
            del lac_2_image_3d


        torch.cuda.empty_cache()   # single cleanup after all GT images saved
    # --------------------------------------------------------------------------



# ==============================================================================
# Class Ends
# ==============================================================================