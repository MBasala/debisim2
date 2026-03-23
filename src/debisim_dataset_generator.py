#!/usr/bin/env python

# -----------------------------------------------------------------------------
"""debisim_dataset_generator.py: Program for generating randomized X-ray datasets
                            using the DEBISim pipeline.
"""

__author__    = "Ankit Manerikar"
__copyright__ = "Copyright (C) 2018, DEBATR Project"
__date__      = "5th January, 2020"
__credits__   = ["Ankit Manerikar", "Fangda Li", "Dr. Tanmay Prakash",
                 "Dr. Avinash Kak"]
__license__   = "Public Domain"
__version__   = "1.2.0"
__maintainer__= ["Ankit Manerikar", "Fangda Li"]
__email__     = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__    = "Prototype"
# -----------------------------------------------------------------------------

import os
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

import torch
from src.debisim_pipeline import DEBISimPipeline
from lib.misc.benchmark import Benchmark
from lib.misc.util import flush_async_io

img_suffixes = dict(
    cdm=['c', 'pe', 'z'],
    sirz=['ze', 'rho']
)

"""
-------------------------------------------------------------------------------
* Module Description:

The module contains the function run_xray_dataset_generator() that executes the 
entire DebiSim simulation pipeline to generate a randomized baggage dataset. 
The data for each of the different blocks of the simulation pipeline are fed as 
input arguments. Running the functions executes each of the pipeline blocks in 
sequence:

1) bag_generator
2) forward_model
3) decomposer
4) reconstructor

The dataset generator runs iterations of the DEBISimPipeline() to generate a 
set of simulation directories one for each data instance. The structure of the 
simulation directory and its contents is shown in DEBISimPipeline(). The 
bag_generator and forward_model blocks are run implicitly within 
DEBISimPipeline(). This is followed by the decomposer block which extracts 
dual energy coefficient data from the projection data pairs. This step is 
optional and is executed only if a decomposer is provided as input. If no 
decomposer is provided, the code proceeds to the reconstructor block. The 
decomposition method(s) (CDM or SIRZ) can be determined by providing the 
decomposer and its corresponding arguments (Both types of reconstructors can be 
provided to the function simultaneously as a list). Once decomposition is 
performed,the reconstructor block reconstructs the volumetric images for the 
single/dual energy projections and saves them. The procedure for creating datasets 
from this module is explained in the next section.

-------------------------------------------------------------------------------
"""

def _process_single_bag(args_tuple):
    """
    Worker function for parallel bag generation.
    Must be module-level (not nested) for pickling with 'spawn' on Windows.

    The last element of args_tuple is a multiprocessing.Queue of GPU device
    IDs.  The worker blocks until a GPU is free, claims it via
    CUDA_VISIBLE_DEVICES, and returns it to the queue when finished so that
    only one worker occupies each physical GPU at a time.
    """
    (*payload, gpu_queue) = args_tuple
    (bag_dir, scanner_init_args, xray_src_mdl, bag_creator_args,
     fwd_mdl_args, decomposer, decomposer_args, basis_fn,
     save_sino, recon_args, images_to_save, slicewise,
     compress_data, dicom_output, compress_dicom) = payload

    # ---- Acquire a free GPU ------------------------------------------------
    gpu_id = gpu_queue.get()          # blocks until a GPU is available
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"[Worker {os.getpid()}] Acquired GPU {gpu_id} for {bag_dir}")

    try:
        # Initialize CUDA context in this worker process.
        # IMPORTANT: these imports are deferred to AFTER CUDA_VISIBLE_DEVICES
        # is set, so that torch binds to the correct GPU.  The module-level
        # `from src.debisim_pipeline import *` may have already imported torch
        # but without initializing CUDA (spawn re-imports the module).
        import torch
        if gpu_id >= 0 and torch.cuda.is_available():
            torch.cuda.init()
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        from src.debisim_pipeline import DEBISimPipeline
        from lib.forward_model.scanner_template import ScannerTemplate
        scanner = ScannerTemplate(
            geometry=scanner_init_args['geometry'],
            scan=scanner_init_args['scan'],
            machine_dict=scanner_init_args['machine_dict'],
            recon=scanner_init_args['recon'],
            recon_dict=scanner_init_args['recon_dict'],
            pscale=scanner_init_args['pscale'],
        )
        scanner.set_recon_geometry()

        # Run the per-bag pipeline (same logic as the sequential `run` closure)
        simulator = DEBISimPipeline(
            sim_path=bag_dir,
            scanner_model=scanner,
            xray_source_model=xray_src_mdl,
            compress_data=compress_data
        )

        simulator.logger.info("=" * 80)
        simulator.logger.info(" " * 20 + "DEBISIM PIPELINE STARTS" + " " * 20)
        simulator.logger.info("=" * 80)

        simulator.mu.material('water')

        simulator.logger.info('\n' + "-" * 50 + "BAG_GENERATOR" + "-" * 50 + '\n')
        simulator.create_random_simulation_instance(bag_creator_args,
                                                    save_images=images_to_save,
                                                    slicewise=slicewise)

        simulator.logger.info('\n' + "-" * 50 + "FORWARD_MODEL" + "-" * 50 + '\n')
        simulator.run_fwd_model(**fwd_mdl_args)

        torch.cuda.empty_cache()

        if decomposer != 'none':
            simulator.logger.info('\n' + '-' * 50 + "DECOMPOSER" + '-' * 50 + '\n')
            simulator.run_decomposer(type=decomposer,
                                     decomposer_args=decomposer_args,
                                     basis_fn=basis_fn,
                                     save_sino=save_sino)

        simulator.logger.info('\n' + "-" * 50 + "RECONSTRUCTOR" + "-" * 50 + '\n')

        if recon_args is not None:
            simulator.run_reconstructor(**recon_args)
        else:
            simulator.run_reconstructor()

        if dicom_output:
            simulator.save_dicom_output()
            if compress_dicom:
                _compress_dicom_dir(simulator.f_loc['image_dir'],
                                    simulator.logger)

        if simulator.logger.hasHandlers():
            simulator.logger.handlers.clear()
        simulator.logger.propagate = False
        simulator.scanner.logger.propagate = False

        del simulator.logger
        del simulator
        torch.cuda.empty_cache()

    finally:
        # ---- Release GPU back to the pool ----------------------------------
        gpu_queue.put(gpu_id)
        print(f"[Worker {os.getpid()}] Released GPU {gpu_id}")
# -----------------------------------------------------------------------------


def _compress_dicom_dir(image_dir, logger=None):
    """Compress the dicom/ subfolder into dicom.zip and remove the original."""
    import shutil
    dicom_dir = os.path.join(image_dir, 'dicom')
    if not os.path.isdir(dicom_dir):
        return
    zip_path = os.path.join(image_dir, 'dicom')  # shutil adds .zip
    shutil.make_archive(zip_path, 'zip', root_dir=image_dir, base_dir='dicom')
    shutil.rmtree(dicom_dir)
    if logger:
        logger.info("DICOM compressed to %s.zip" % zip_path)


def run_xray_dataset_generator(num_bags,
                               sim_dir,
                               scanner,
                               xray_src_mdl,
                               bag_creator_args,
                               decomposer='cdm',
                               save_sino=False,
                               basis_fn=None,
                               decomposer_args=None,
                               recon_args=None,
                               images_to_save=None,
                               slicewise=True,
                               compress_data=False,
                               fwd_mdl_args=None,
                               num_workers=1,
                               dicom_output=False,
                               compress_dicom=False
):
    """
    ---------------------------------------------------------------------------
    Function to generate a CT baggage Dataset using the DebiSim pipeline.

    :param num_bags:            number of baggage instances to generate
    :param sim_dir:             simulation directory
    :param scanner:             scanner model defined from ScannerTemplate()
    :param xray_src_mdl:        Xray source specifications - these are defined
                                as a dictionary with the following
                                specifications:
                                - 'num_spectra' - No of X-ray sources/spectra
                                - 'kVp' - peak kV voltage for the X-ray source(s)
                                - 'spectra' - file paths for the each of the
                                              X-ray spectra. The spectrum files
                                              must contain a N x 2 array with the
                                              keV values in the first column
                                              and normalized photon distribution
                                              in the 2nd column. See
                                              /include/spectra/ for reference.
                                - 'dosage' - dosage count for each of the sources
    :param bag_creator_args:    The baggage creation arguments - this dictionary
                                contains the input arguments to the method
                                self.create_random_object_list() for
                                BaggageCreator3D() or BaggageCreator2d()
    :param decomposer:          The object(s) corresponding the DE Decomposer,
                                for e.g., CDMDecomposer().
    :param decomposer_args:     Input arguments for the specified decomposer(s)
    :param save_sino:           If simulated sinograms are to be saved
    :param basis_fn:            Energy basis functions for DECT processing
    :param images_to_save:      ground truth images to be saved:
                                {gt | lac_1 | lac_2 | compton | pe | zeff}
    :param slicewise:           whether to produce H x W X D 3D volumetric bag
                                or H x W 2D baggage cross-sections in D batches
    :param compress_data:       whether to compress saved FITS data
    :param fws_model_args       arguments for X-ray forward model

    :return:
    ---------------------------------------------------------------------------
    """

    if images_to_save is None:
        images_to_save = ['lac_1', 'pe', 'c', 'gt']

    if fwd_mdl_args is None:
        fwd_mdl_args = dict(
            add_poisson_noise=True,
            add_system_noise=True,
            system_gain=0.025
        )

    sim_bag_dir = os.path.join(sim_dir, 'simulation_%03d/')

    # --- Parallel path ---------------------------------------------------
    if num_workers > 1:
        # Extract picklable scanner construction args (the live scanner
        # object contains a logger and bound methods that can't be pickled)
        scanner_init_args = dict(
            geometry=scanner.geom,
            scan=scanner.scan,
            machine_dict=scanner.machine_geometry,
            recon=scanner.recon,
            recon_dict=scanner.recon_params,
            pscale=scanner.pscale,
        )

        bag_dirs = [sim_bag_dir % i for i in num_bags]
        for d in bag_dirs:
            os.makedirs(d, exist_ok=True)

        # ---- GPU-aware scheduling ----------------------------------------
        # Detect available GPUs.  Each GPU ID is placed in a shared queue.
        # Workers block on queue.get() until a GPU is free, and return it
        # via queue.put() when done.  This guarantees at most one worker
        # per physical GPU at any time.
        import torch as _torch
        num_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
        del _torch

        ctx = mp.get_context('spawn')
        mgr = ctx.Manager()
        gpu_queue = mgr.Queue()

        if num_gpus > 0:
            for gid in range(num_gpus):
                gpu_queue.put(gid)
            effective_workers = min(num_workers, num_gpus)
            print(f"Detected {num_gpus} GPU(s). "
                  f"Capping parallel workers to {effective_workers}.")
        else:
            # No GPUs — let every worker run on CPU (put sentinel -1)
            for _ in range(num_workers):
                gpu_queue.put(-1)
            effective_workers = num_workers
            print("No GPUs detected — running all workers on CPU.")
        # ------------------------------------------------------------------

        work_items = [
            (bag_dir, scanner_init_args, xray_src_mdl, bag_creator_args,
             fwd_mdl_args, decomposer, decomposer_args, basis_fn,
             save_sino, recon_args, images_to_save, slicewise,
             compress_data, dicom_output, compress_dicom, gpu_queue)
            for bag_dir in bag_dirs
        ]

        print(f"Starting parallel generation of {len(bag_dirs)} bags "
              f"with {effective_workers} workers ...")

        with ProcessPoolExecutor(max_workers=effective_workers,
                                 mp_context=ctx) as executor:
            futures = {executor.submit(_process_single_bag, item): item[0]
                       for item in work_items}
            for future in as_completed(futures):
                bag_dir = futures[future]
                try:
                    future.result()
                    print(f'Completed: {bag_dir}')
                except Exception as e:
                    print(f'Failed: {bag_dir} with error: {e}')

        mgr.shutdown()
        print("All bags processed.")
        return
    # --- End parallel path -----------------------------------------------

    # Organize processing with Benchmark (sequential path)
    bench = Benchmark(save_log=False,
                      save_remark=False)
    bench.set_remark(
        'Creating Randomized DEBISIM Dataset %s with %i bags ....'%(sim_dir,
                                                            len(num_bags)))
    bench.set_test_cases([sim_bag_dir % i for i in num_bags])
    bench.set_output_dir([sim_bag_dir % i for i in num_bags])

    # Process slice by slice
    def preprocess(test_case):
        # ---------------------------------------------------------------------
        # Setup the simulator
        # ---------------------------------------------------------------------

        pre = dict(test_case=test_case)
        return pre
        # ---------------------------------------------------------------------

    def run(pre):
        # ---------------------------------------------------------------------
        # Run the simulator
        # ---------------------------------------------------------------------
        bag_dir = pre['test_case']

        # Initialized the simulator
        simulator = DEBISimPipeline(
            sim_path=bag_dir,
            scanner_model=scanner,
            xray_source_model=xray_src_mdl,
            compress_data=compress_data
            # zwidth=scanner.recon_params['image_dims'][2]
        )

        simulator.logger.info("="*80)
        simulator.logger.info(" "*20+"DEBISIM PIPELINE STARTS"+" "*20)
        simulator.logger.info("="*80)

        mu = simulator.mu.material('water')

        simulator.logger.info('\n'+"-" * 50+"BAG_GENERATOR"+"-" * 50+'\n')
        simulator.create_random_simulation_instance(bag_creator_args,
                                                    save_images=images_to_save,
                                                    slicewise=slicewise)

        # simulator.keV_range = range(50,55)

        simulator.logger.info('\n'+"-" * 50+"FORWARD_MODEL"+"-" * 50+'\n')

        simulator.run_fwd_model(**fwd_mdl_args)

        f_loc = simulator.f_loc.copy()

        if decomposer=='none':
            pass
        else:

            simulator.logger.info('\n'+'-'*50+"DECOMPOSER"+'-'*50+'\n')

            simulator.run_decomposer(type=decomposer,
                                     decomposer_args=decomposer_args,
                                     basis_fn=basis_fn,
                                     save_sino=save_sino)

        simulator.logger.info('\n'+"-"*50+"RECONSTRUCTOR"+"-"*50+'\n')

        if recon_args is not None:
            simulator.run_reconstructor(**recon_args)
        else:
            simulator.run_reconstructor()

        if dicom_output:
            simulator.save_dicom_output()
            if compress_dicom:
                _compress_dicom_dir(simulator.f_loc['image_dir'],
                                    simulator.logger)

        res = dict()

        if (simulator.logger.hasHandlers()):
            simulator.logger.handlers.clear()

        simulator.logger.propagate = False
        simulator.scanner.logger.propagate = False

        del simulator.logger
        del simulator

        return res
    # -------------------------------------------------------------------------

    def postprocess(res, outdir):

        if res is not None:
            for k, v in res.items():
                del v

        del res
        return None

    bench.set_handles(preprocess, run, postprocess, None)
    bench.start()
# -----------------------------------------------------------------------------

