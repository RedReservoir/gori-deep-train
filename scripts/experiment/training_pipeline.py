import os
import sys
import shutil
import argparse
import logging
import datetime
import traceback
import importlib
import random

import numpy

import torch

####

from gorideep.utils.errors import AbortExperimentError

from gorideep.loss_registers.step_wise import StepWiseLossRegister
from gorideep.loss_registers.epoch_wise import EpochWiseLossRegister

####

import goripy.file.json
import goripy.log
import goripy.tqdm
import goripy.mldl.multibatch
import goripy.gpu.info



###########
# UTILITIES
###########



class OnlyRankZero:
    """
    A context manager for operations in the pipeline that are only supposed to be performed by the
    rank 0 subprocess. Provides additional logging and unexpected exception handling.

    If the rank 0 subprocess raises an exception, it will be caught and all subprocesses will raise
    an AbortExperimentError, ensuring all of them are terminated together.

    When running code inside this context manager, use the `is_rank_zero` method to check whether
    the current process is actually the rank 0 subprocess.
    """
    
    def __init__(
        self,
        rank,
        logger
    ):

        self._rank = rank
        self._logger = logger

        self._status_dict_ptdl = [{
            "abort": False,
            "traceback": None
        }]


    def __enter__(
        self
    ):

        if self._rank != 0:
            self._logger.info("Waiting for rank 0 subprocess...")

        return self


    def __exit__(
        self,
        exc_type,
        exc_value,
        exc_traceback
    ):

        if self._rank == 0 and exc_type is not None:

            self._status_dict_ptdl = [{
                "abort": True,
                "traceback": traceback.format_exc()
            }]

        torch.distributed.broadcast_object_list(self._status_dict_ptdl, src=0)
        torch.distributed.barrier(device_ids=[self._rank])

        if self._rank != 0:
            self._logger.info("... rank 0 subprocess finished")

        if self._status_dict_ptdl[0]["abort"]:

            raise AbortExperimentError(
                message="Experiment aborted",
                orig_traceback=self._status_dict_ptdl[0]["traceback"]
            )

        return True
    

    def is_rank_zero(
        self
    ):

        return self._rank == 0



##################
# EXPERIMENT LOOPS
##################



def training_loop(
    command_args,
    exp_data,

    data_loading_settings,
    logging_settings,

    dataset_pool,
    dataset_name_list,
    data_transform_pool,
    module_pool,
    optimizer_pool,
    scheduler_pool,
    module_transform_pool,
    module_transform_loss_ten_reg_key_list_dict,
    loss_reg_pool,
    loss_weighter,

    logger,
    tqdm_logger,
):


    #
    # Device initialization
    #


    rank = torch.distributed.get_rank()
    device = torch.device(rank)


    #
    # Build dataloaders and metadata
    #


    logger.info("Building dataloaders")

    dataloader_iter_pool = {}

    dataloader_len_list = []
    dataloader_batch_size_list = []
    dataset_point_size_list = []

    split_dataset_name_list = []

    for dataset_name in dataset_name_list:       

        # Prepare dataset

        dataset = dataset_pool[dataset_name]
        
        split_idxs = dataset.get_split_idxs("train")
        if len(split_idxs) == 0: continue

        data_transform = data_transform_pool["train"][dataset_name]
        dataset.set_data_transform(data_transform)

        split_dataset = torch.utils.data.Subset(dataset, split_idxs)

        # Prepare dataloader

        dataloader_pymodule_name = "dataloaders.{:s}".format(dataset_name)
        dataloader_pymodule = importlib.import_module(dataloader_pymodule_name)

        dataloader_collate_fn = dataloader_pymodule.dataloader_collate_fn

        dataloader_batch_size = dataloader_pymodule.train_dataloader_args["batch_size"]
        dataloader_num_workers = dataloader_pymodule.train_dataloader_args["num_workers"]
        dataloader_prefetch_factor = dataloader_pymodule.train_dataloader_args["prefetch_factor"]
        dataloader_point_size = dataloader_pymodule.train_dataloader_args["point_size"]

        dist_sampler = torch.utils.data.DistributedSampler(
            dataset=split_dataset,
            shuffle=True
        )

        dist_sampler.set_epoch(exp_data["status"]["pipeline_state"]["epoch_num"])

        split_dataloader = torch.utils.data.DataLoader(
            dataset=split_dataset,
            batch_size=dataloader_batch_size,
            sampler=dist_sampler,
            num_workers=dataloader_num_workers,
            collate_fn=dataloader_collate_fn,
            pin_memory=True,
            prefetch_factor=dataloader_prefetch_factor,
        )

        # Store dataset, dataloader and metadata

        dataloader_iter_pool[dataset_name] = iter(split_dataloader)

        dataloader_len_list.append(len(split_dataloader))
        dataloader_batch_size_list.append(dataloader_batch_size)
        dataset_point_size_list.append(dataloader_point_size)

        split_dataset_name_list.append(dataset_name)


    #
    # Pre-compute batches per step
    #


    logger.info("Pre-computing batches per step")
    
    dataloader_len_arr = numpy.asarray(dataloader_len_list, dtype=int)
    dataloader_batch_size_arr = numpy.asarray(dataloader_batch_size_list, dtype=int)
    dataset_point_size_arr = numpy.asarray(dataset_point_size_list, dtype=int)

    step_num_batches_arrr = goripy.mldl.multibatch.compute_multibatches_dataloader(
        dataloader_len_arr,
        dataloader_batch_size_arr,
        dataset_point_size_arr,
        data_loading_settings["step_min_items"],
        world_size
    )

    epoch_num_steps = len(step_num_batches_arrr)


    #
    # Before epoch operations
    #


    logger.info("Before epoch operations")
    
    for module_name, module in module_pool.items():
        module.train()

    for loss_reg_name, loss_reg in loss_reg_pool["train"].items():
        loss_reg.initialize_step_data(len(step_num_batches_arrr))

    loss_weighter.event_before_train_epoch(
        loss_reg_pool
    )
    
    for scheduler_name, scheduler in scheduler_pool.items():
        scheduler.event_before_train_epoch(
            epoch_num_steps
        )


    #
    # Iterate through steps
    #


    logger.info("Iterating through steps...")

    nan_loss_detected_sync_ten = torch.empty(size=(1,), dtype=bool, device=device)
    nan_loss_detected_flag = None

    step_num_batches_arr_gen = step_num_batches_arrr
    
    if logging_settings["tqdm"]["enabled"]:
        step_num_batches_arr_gen = goripy.tqdm.tqdmidify(
            step_num_batches_arr_gen,
            tqdm_len=step_num_batches_arrr.shape[0],
            tqdm_freq=logging_settings["tqdm"]["train_freq"],
            tqdm_file=tqdm_logger
        )

    for epoch_step_idx, step_num_batches_arr in enumerate(step_num_batches_arr_gen):

        # Step preparation

        nan_loss_detected_sync_ten[0] = False
        nan_loss_detected_flag = False

        step_unique_dataset_name_list = []
        step_dataset_name_batch_idx_tuple_list = []

        for dataset_name, num_batches in zip(split_dataset_name_list, step_num_batches_arr):
            if num_batches == 0: continue
            step_unique_dataset_name_list.append(dataset_name)
            step_dataset_name_batch_idx_tuple_list += \
                [(dataset_name, batch_idx) for batch_idx in range(num_batches)]

        random.shuffle(step_dataset_name_batch_idx_tuple_list)

        # Process step batches

        for batch_dataset_name, dataset_batch_idx in step_dataset_name_batch_idx_tuple_list:

            ## Get next batch from dataloader

            data_batch = next(dataloader_iter_pool[batch_dataset_name])

            ## Skip all other operations if NaN losses were detected

            if nan_loss_detected_flag:
                continue

            loss_ten_reg_key_list = module_transform_loss_ten_reg_key_list_dict[batch_dataset_name]

            ## Forward pass

            for data_batch_key, data_batch_value in data_batch.items():
                if type(data_batch_value) is torch.Tensor:
                    data_batch[data_batch_key] = data_batch_value.to(device)

            module_transform = module_transform_pool[batch_dataset_name]
            module_transform(data_batch, module_pool)

            ## Check for NaN losses, and stop if detected

            for loss_ten_reg_key in loss_ten_reg_key_list:

                loss_ten_key = loss_ten_reg_key["loss_ten_key"]
                loss_reg_key = loss_ten_reg_key["loss_reg_key"]

                loss_ten = data_batch[loss_ten_key]

                if torch.isnan(loss_ten).any().item():

                    nan_loss_detected_sync_ten[0] = True
                    nan_loss_detected_flag = True

                    log_msg = "Detected NaN loss"
                    log_msg += " - "
                    log_msg += "Dataset: {:s}".format(batch_dataset_name)
                    log_msg += " - "
                    log_msg += "Batch index: {:d}".format(dataset_batch_idx)
                    log_msg += " - "
                    log_msg += "Loss register: {:s}".format(loss_reg_key)
                    logger.debug(log_msg)

                    break
                
            ## Synchronize NaN loss detected flag

            with torch.no_grad():
                torch.distributed.all_reduce(nan_loss_detected_sync_ten, torch.distributed.ReduceOp.MAX)
            
            nan_loss_detected_flag = nan_loss_detected_sync_ten.item()

            ## Skip all other operations if NaN losses were detected

            if nan_loss_detected_flag:
                del data_batch
                continue

            ## Apply loss weights

            for loss_ten_reg_key in loss_ten_reg_key_list:

                loss_ten_key = loss_ten_reg_key["loss_ten_key"]
                loss_reg_key = loss_ten_reg_key["loss_reg_key"]

                loss_ten = data_batch[loss_ten_key]
                loss_weight = loss_weighter.get_loss_weight(loss_reg_key)

                loss_ten *= loss_weight

            ## Register losses and backward pass

            with torch.no_grad():

                for loss_ten_reg_key in loss_ten_reg_key_list:

                    loss_ten_key = loss_ten_reg_key["loss_ten_key"]
                    loss_reg_key = loss_ten_reg_key["loss_reg_key"]

                    loss_ten = data_batch[loss_ten_key]
                    loss_reg = loss_reg_pool["train"][loss_reg_key]

                    loss_reg.accumulate_batch(
                        torch.sum(loss_ten).item(),
                        loss_ten.shape[0]
                    )

            ## Backward pass

            loss_ten_key_list = [
                loss_ten_reg_key["loss_ten_key"]
                for loss_ten_reg_key in loss_ten_reg_key_list
            ]

            loss_ten_list = [
                data_batch[loss_ten_key]
                for loss_ten_key in loss_ten_key_list
            ]

            total_loss_ten = torch.stack(tuple(
                torch.sum(loss_ten)
                for loss_ten in loss_ten_list
            )).sum()
            
            total_loss_ten.backward()
                
        # After step opterations

        if nan_loss_detected_flag:
            logger.debug("Detected NaN loss - Skipping step")

        if nan_loss_detected_flag:
            for dataset_name in step_unique_dataset_name_list:
                loss_ten_reg_key_list = module_transform_loss_ten_reg_key_list_dict[dataset_name]
                for loss_ten_reg_key in loss_ten_reg_key_list:
                    loss_reg_key = loss_ten_reg_key["loss_reg_key"]
                    loss_reg = loss_reg_pool["train"][loss_reg_key]
                    loss_reg.mark_nan_step()

        for loss_reg_name, loss_reg in loss_reg_pool["train"].items():
            loss_reg.store_curr_step_data()

        loss_weighter.event_after_train_step(
            loss_reg_pool
        )

        for optimizer_name, optimizer in optimizer_pool.items():
            if not nan_loss_detected_flag: optimizer.step()
            optimizer.zero_grad()

        for optim_sched_name, scheduler in scheduler_pool.items():
            scheduler.event_after_train_step(epoch_step_idx)            
            
        torch.distributed.barrier(device_ids=[rank])


    #
    # After epoch operations
    #

    
    logger.info("After epoch operations")

    for loss_reg_key, loss_reg in loss_reg_pool["train"].items():
        loss_reg.synchronize_epoch_data()
        loss_reg.store_curr_epoch_data()

    loss_weighter.event_after_train_epoch(loss_reg_pool)

    for optim_sched_name, scheduler in scheduler_pool.items():
        scheduler.event_after_train_epoch()


    #
    # Report GPU memory utilization
    #


    logger.info(goripy.gpu.info.sprint_device_memory_usage(device))   



def evaluation_loop(
    command_args,
    exp_data,

    data_loading_settings,
    logging_settings,

    dataset_pool,
    dataset_name_list,
    data_transform_pool,
    module_pool,
    module_transform_pool,
    module_transform_loss_ten_reg_key_list_dict,
    loss_reg_pool,
    loss_weighter,

    logger,
    tqdm_logger,
):


    #
    # Device initialization
    #


    rank = torch.distributed.get_rank()
    device = torch.device(rank)
    

    #
    # Build dataloaders
    #


    logger.info("Building dataloaders")

    dataloader_pool = {}

    for dataset_name in dataset_name_list:

        # Prepare dataset

        dataset = dataset_pool[dataset_name]
        
        split_idxs = dataset.get_split_idxs("val")
        if len(split_idxs) == 0: continue
        
        data_transform = data_transform_pool["eval"][dataset_name]
        dataset.set_data_transform(data_transform)

        split_dataset = torch.utils.data.Subset(dataset, split_idxs)

        # Prepare dataloader

        dataloader_pymodule_name = "dataloaders.{:s}".format(dataset_name)
        dataloader_pymodule = importlib.import_module(dataloader_pymodule_name)

        dataloader_collate_fn = dataloader_pymodule.dataloader_collate_fn

        dataloader_batch_size = dataloader_pymodule.eval_dataloader_args["batch_size"]
        dataloader_num_workers = dataloader_pymodule.eval_dataloader_args["num_workers"]
        dataloader_prefetch_factor = dataloader_pymodule.eval_dataloader_args["prefetch_factor"]

        dist_sampler = torch.utils.data.DistributedSampler(
            dataset=split_dataset,
            shuffle=True
        )

        dist_sampler.set_epoch(exp_data["status"]["pipeline_state"]["epoch_num"])

        split_dataloader = torch.utils.data.DataLoader(
            dataset=split_dataset,
            batch_size=dataloader_batch_size,
            sampler=dist_sampler,
            num_workers=dataloader_num_workers,
            collate_fn=dataloader_collate_fn,
            pin_memory=True,
            prefetch_factor=dataloader_prefetch_factor,
        )

        # Store dataloader

        dataloader_pool[dataset_name] = split_dataloader


    #
    # Before epoch operations
    #


    logger.info("Before epoch operations")
    
    for module_name, module in module_pool.items():
        module.eval()

    for loss_reg_name, loss_reg in loss_reg_pool["val"].items():
        loss_reg.initialize_epoch_data()

    loss_weighter.event_before_val_epoch(
        loss_reg_pool
    )
    

    #
    # Iterate through dataset batches
    #


    logger.info("Iterating through dataset batches...")

    for dataset_name, dataloader in dataloader_pool.items():

        # Dataset preparation

        loss_ten_reg_key_list = module_transform_loss_ten_reg_key_list_dict[dataset_name]

        logger.info("Dataset: {:s}".format(dataset_name))

        if logging_settings["tqdm"]["enabled"]:
            dataloader = goripy.tqdm.tqdmidify(
                dataloader,
                tqdm_len=len(dataloader),
                tqdm_freq=logging_settings["tqdm"]["train_freq"],
                tqdm_file=tqdm_logger
            )

        # Process dataset batches

        for data_batch in dataloader:

            with torch.no_grad():

                ## Forward pass

                for data_batch_key, data_batch_value in data_batch.items():
                    if type(data_batch_value) is torch.Tensor:
                        data_batch[data_batch_key] = data_batch_value.to(device)

                module_transform = module_transform_pool[dataset_name]
                module_transform(data_batch, module_pool)

                ## Apply loss weights

                for loss_ten_reg_key in loss_ten_reg_key_list:

                    loss_ten_key = loss_ten_reg_key["loss_ten_key"]
                    loss_reg_key = loss_ten_reg_key["loss_reg_key"]

                    loss_ten = data_batch[loss_ten_key]
                    loss_weight = loss_weighter.get_loss_weight(loss_reg_key)

                    loss_ten *= loss_weight

                ## Register losses

                for loss_ten_reg_key in loss_ten_reg_key_list:

                    loss_ten_key = loss_ten_reg_key["loss_ten_key"]
                    loss_reg_key = loss_ten_reg_key["loss_reg_key"]

                    loss_ten = data_batch[loss_ten_key]
                    loss_reg = loss_reg_pool["val"][loss_reg_key]

                    if torch.isnan(loss_ten).any().item():

                        log_msg = "Detected NaN loss"
                        log_msg += " - "
                        log_msg += "Loss: {:s}".format(loss_ten_key)
                        logger.debug(log_msg)

                        loss_reg.accumulate_nan_batch()
                    
                    else:

                        loss_reg.accumulate_batch(
                            torch.sum(loss_ten).item(),
                            loss_ten.shape[0]
                        )

    torch.distributed.barrier(device_ids=[rank])


    #
    # After epoch operations
    #


    logger.info("After epoch operations")

    for loss_reg_key, loss_reg in loss_reg_pool["val"].items():
        loss_reg.synchronize_epoch_data()
        loss_reg.store_curr_epoch_data()

    loss_weighter.event_after_val_epoch(
        loss_reg_pool
    )


    #
    # Repor GPU memory utilizatiotn
    #


    logger.info(goripy.gpu.info.sprint_device_memory_usage(device))   



#####################
# EXPERIMENT PIPELINE
#####################



def experiment_pipeline():


    #
    # Device initialization
    #


    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(rank)


    #
    # Capture current time
    #


    run_datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    

    #
    # Parse command args
    #


    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "exp_name",
        help="name of the experiment run"
    )
    
    parser.add_argument(
        "--resume",
        help="resumes experiment from current state",
        action="store_true"
    )
    
    parser.add_argument(
        "--reset",
        help="resets experiment and starts anew",
        action="store_true"
    )
    
    parser.add_argument(
        "--autograd_anomaly",
        help="use torch.autograd.set_detect_anomaly(True) for debugging",
        action="store_true"
    )
    
    command_args = parser.parse_args()
    
    if command_args.autograd_anomaly:
        torch.autograd.set_detect_anomaly(True)


    #
    # Setup logging
    #


    # Create logger

    logger_name = "rank {:d}".format(local_rank)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter("%(name)10s > %(asctime)s %(levelname)8s: %(message)s")

    # Create stdout handler

    log_stdout_handler = logging.StreamHandler(sys.stdout)
    log_stdout_handler.setFormatter(log_formatter)

    logger.addHandler(log_stdout_handler)

    # Create tqdm logger

    tqdm_logger = goripy.log.TqdmLogger(
        logger,
        log_level=logging.DEBUG
    )


    #
    # Check command argument flags and create experiment directories
    #


    exp_settings_dirname = os.path.join(os.environ["GORIDEEPTRAIN_DATA_HOME"], "experiment_settings", command_args.exp_name)
    exp_results_dirname = os.path.join(os.environ["GORIDEEPTRAIN_DATA_HOME"], "experiment_results", command_args.exp_name)
    #exp_logs_dirname = os.path.join(os.environ["GORIDEEPTRAIN_DATA_HOME"], "experiment_logs", command_args.exp_name)

    exp_settings_settings_dirname = os.path.join(exp_settings_dirname, "settings")
    exp_settings_pymodules_dirname = os.path.join(exp_settings_dirname, "pymodules")


    with OnlyRankZero(rank, logger) as only_rank_zero:
        if only_rank_zero.is_rank_zero():

            if command_args.resume and command_args.reset:

                raise ValueError("Flags --resume and --reset cannot be active simultaneously")

            elif command_args.resume:

                if not os.path.exists(exp_results_dirname):
                    raise ValueError("Experiment results directory does not exist but --resume flag is enabled")

            elif command_args.reset:

                if os.path.exists(exp_results_dirname): shutil.rmtree(exp_results_dirname)
                os.makedirs(exp_results_dirname, exist_ok=True)
            
            else:

                if os.path.exists(exp_results_dirname):
                    raise ValueError("Experiment results directory already exists, provide the --resume or the --reset flag")
                
                os.makedirs(exp_results_dirname, exist_ok=True)


    #
    # Prepare experiment data
    #

    
    exp_data_filename = os.path.join(exp_results_dirname, "exp_data.json")
    
    # Create or update experiment data

    with OnlyRankZero(rank, logger) as only_rank_zero:
        if only_rank_zero.is_rank_zero():

            if os.path.exists(exp_data_filename):
            
                logger.info("Update experiment data")
                
                exp_data = goripy.file.json.load_json(exp_data_filename)
                exp_data["status"]["run_datetime_list"].append(run_datetime_str)
            
            else:

                logger.info("Create experiment data")

                exp_data = {}

                exp_data["metadata"] = {}

                exp_data["status"] = {
                    "run_datetime_list": [run_datetime_str],
                    "requirements": {
                        "exp_results_settings_ready": False,
                        "exp_results_pymodules_ready": False,
                        "exp_results_data_ready": False,
                        "data_counters_ready": False,
                        "module_ckps_ready": False,
                        "optim_sched_data_ready": False,
                        "loss_reg_data_ready": False,
                        "loss_weighter_data_ready": False,
                        "early_stopper_data_ready": False,
                        "checkpoint_saver_data_ready": False,
                    },
                    "pipeline_state": {
                        "epoch_num": 0,
                        "stage_name": "epoch_0_evaluation_loop",
                        "temp_data_cleanup_list": []
                    }
                }

            goripy.file.json.save_json(exp_data, exp_data_filename)

    # Load experiment data and show current progress

    logger.info("Load experiment data")
    exp_data = goripy.file.json.load_json(exp_data_filename)

    logger.info("Requirements:")
    for req_key, req_value in exp_data["status"]["requirements"].items():
        logger.info("  {:30s} {:>5s}".format(req_key, str(req_value)))
    logger.info("Pipeline state:")
    logger.info("  epoch {:d}".format(exp_data["status"]["pipeline_state"]["epoch_num"]))
    logger.info("  stage {:s}".format(exp_data["status"]["pipeline_state"]["stage_name"]))


    #
    # Define experiment result directories
    #

    
    exp_results_settings_dirname = os.path.join(exp_results_dirname, "settings")    
    exp_results_pymodules_dirname = os.path.join(exp_results_dirname, "pymodules")
    exp_results_data_dirname = os.path.join(exp_results_dirname, "data")
    exp_results_temp_data_dirname = os.path.join(exp_results_dirname, "temp_data")


    #
    # Prepare experiment results settings
    #
    

    if not exp_data["status"]["requirements"]["exp_results_settings_ready"]:

        with OnlyRankZero(rank, logger) as only_rank_zero:
            if only_rank_zero.is_rank_zero():

                logger.info("Prepare experiment results settings")

                ## Create experiment settings directory

                if os.path.exists(exp_results_settings_dirname): shutil.rmtree(exp_results_settings_dirname)
                os.mkdir(exp_results_settings_dirname)

                ## Copy experiment settings
                
                for settings_filename in [
                    "logging.json",
                    "loss_registers.json",
                    "data_loading.json"
                ]:
                    
                    settings_src_full_filename = os.path.join(exp_settings_settings_dirname, settings_filename)
                    settings_dst_full_filename = os.path.join(exp_results_settings_dirname, settings_filename)

                    if not os.path.exists(settings_src_full_filename):
                        raise ValueError("Missing settings file: {:s}".format(settings_src_full_filename))

                    shutil.copyfile(settings_src_full_filename, settings_dst_full_filename)

                ## Update experiment settings

                logger.info("Update experiment data")
                exp_data["status"]["requirements"]["exp_results_settings_ready"] = True
                goripy.file.json.save_json(exp_data, exp_data_filename)

        # Reload experiment data

        logger.info("Reload experiment data")
        exp_data = goripy.file.json.load_json(exp_data_filename)


    #
    # Prepare experiment results pymodules
    #
    

    if not exp_data["status"]["requirements"]["exp_results_pymodules_ready"]:

        with OnlyRankZero(rank, logger) as only_rank_zero:
            if only_rank_zero.is_rank_zero():

                logger.info("Prepare experiment results pymodules")

                ## Create experiment pymodules directory

                if os.path.exists(exp_results_pymodules_dirname): shutil.rmtree(exp_results_pymodules_dirname)
                os.mkdir(exp_results_pymodules_dirname)

                ## Copy experiment pymodules
                
                for pymodules_dirname in [
                    "data_counters",
                    "data_transforms",
                    "dataloaders",
                    "datasets",
                    "module_transforms",
                    "modules",
                    "optim_sched"
                ]:
                    
                    pymodules_src_full_dirname = os.path.join(exp_settings_pymodules_dirname, pymodules_dirname)
                    pymodules_dst_full_dirname = os.path.join(exp_results_pymodules_dirname, pymodules_dirname)

                    if not os.path.exists(pymodules_src_full_dirname):
                        raise ValueError("Missing pymodules directory: {:s}".format(settings_src_full_filename))
                    
                    shutil.copytree(pymodules_src_full_dirname, pymodules_dst_full_dirname)

                    pymodules_init_full_filename = os.path.join(pymodules_dst_full_dirname, "__init__.py")
                    open(pymodules_init_full_filename, "w").close()

                for pymodule_filename in [
                    "checkpoint_saver.py",
                    "early_stopper.py",
                    "loss_weighter.py"
                ]:
                    
                    pymodule_src_full_filename = os.path.join(exp_settings_pymodules_dirname, pymodule_filename)
                    pymodule_dst_full_filename = os.path.join(exp_results_pymodules_dirname, pymodule_filename)

                    if not os.path.exists(pymodule_src_full_filename):
                        raise ValueError("Missing pymodules file: {:s}".format(pymodule_src_full_filename))

                    shutil.copyfile(pymodule_src_full_filename, pymodule_dst_full_filename)

                ## Update experiment data

                logger.info("Update experiment data")
                exp_data["status"]["requirements"]["exp_results_pymodules_ready"] = True
                goripy.file.json.save_json(exp_data, exp_data_filename)

        # Reload experiment data

        logger.info("Reload experiment data")
        exp_data = goripy.file.json.load_json(exp_data_filename)

    # Add experiment pymodules root path to sys.path

    sys.path.insert(0, exp_results_pymodules_dirname)


    #
    # Prepare experiment results data
    #


    if not exp_data["status"]["requirements"]["exp_results_data_ready"]:

        with OnlyRankZero(rank, logger) as only_rank_zero:
            if only_rank_zero.is_rank_zero():

                logger.info("Create experiment results data directories")

                ## Create experiment results data directories

                if os.path.exists(exp_results_data_dirname): shutil.rmtree(exp_results_data_dirname)
                os.mkdir(exp_results_data_dirname)

                if os.path.exists(exp_results_temp_data_dirname): shutil.rmtree(exp_results_temp_data_dirname)
                os.mkdir(exp_results_temp_data_dirname)

                ## Update experiment data

                logger.info("Update experiment data")
                exp_data["status"]["requirements"]["exp_results_data_ready"] = True
                goripy.file.json.save_json(exp_data, exp_data_filename)

        # Reload experiment data

        logger.info("Reload experiment data")
        exp_data = goripy.file.json.load_json(exp_data_filename) 


    #
    # Perform experiment result temp data cleanup
    #


    if len(exp_data["status"]["pipeline_state"]["temp_data_cleanup_list"]) > 0:

        with OnlyRankZero(rank, logger) as only_rank_zero:
            if only_rank_zero.is_rank_zero():

                logger.info("Perform experiment result temp data cleanup")

                ## Perform experiment result temp data cleanup

                for cleanup_data_name in exp_data["status"]["pipeline_state"]["temp_data_cleanup_list"]:

                    full_data_name = os.path.join(exp_results_data_dirname, cleanup_data_name)
                    temp_full_data_name = os.path.join(exp_results_temp_data_dirname, cleanup_data_name)

                    if os.path.exists(temp_full_data_name):

                        if os.path.isfile(temp_full_data_name):

                            if os.path.exists(full_data_name): os.remove(full_data_name)
                            shutil.copyfile(temp_full_data_name, full_data_name)

                        if os.path.isdir(temp_full_data_name):

                            if os.path.exists(full_data_name): shutil.rmtree(full_data_name)
                            shutil.copytree(temp_full_data_name, full_data_name)

                        logger.info("Processed temp data item \"{:s}\"".format(
                            cleanup_data_name
                        ))

                ## Update experiment data

                logger.info("Update experiment data")
                exp_data["status"]["pipeline_state"]["temp_data_cleanup_list"] = []
                goripy.file.json.save_json(exp_data, exp_data_filename)

        # Reload experiment data

        logger.info("Reload experiment data")
        exp_data = goripy.file.json.load_json(exp_data_filename) 


    #
    # Load logging settings
    #


    logging_settings_dirname = os.path.join(exp_results_settings_dirname, "logging.json")
    logging_settings = goripy.file.json.load_json(logging_settings_dirname)


    #
    # Load data loading settings
    #


    data_loading_settings_dirname = os.path.join(exp_results_settings_dirname, "data_loading.json")
    data_loading_settings = goripy.file.json.load_json(data_loading_settings_dirname)


    #
    # Build datasets
    #


    logger.info("Build datasets")

    dataset_pool = {}

    dataset_pymodules_dirname = os.path.join(exp_results_pymodules_dirname, "datasets")
    dataset_name_list = [filename.split(".")[0] for filename in os.listdir(dataset_pymodules_dirname)]
    if "__init__" in dataset_name_list: dataset_name_list.remove("__init__")
    if "__pycache__" in dataset_name_list: dataset_name_list.remove("__pycache__")

    for dataset_name in dataset_name_list:

        dataset_pymodule_name = "datasets.{:s}".format(dataset_name)
        dataset_pymodule = importlib.import_module(dataset_pymodule_name)

        dataset_pool[dataset_name] = dataset_pymodule.dataset


    #
    # Build data counters
    #


    logger.info("Build data counters")

    data_counter_pool = {}
    data_counter_dataset_name_list_dict = {}

    data_counter_pymodules_dirname = os.path.join(exp_results_pymodules_dirname, "data_counters")
    data_counter_name_list = [filename.split(".")[0] for filename in os.listdir(data_counter_pymodules_dirname)]
    if "__init__" in data_counter_name_list: data_counter_name_list.remove("__init__")
    if "__pycache__" in data_counter_name_list: data_counter_name_list.remove("__pycache__")

    for data_counter_name in data_counter_name_list:

        data_counter_pymodule_name = "data_counters.{:s}".format(data_counter_name)
        data_counter_pymodule = importlib.import_module(data_counter_pymodule_name)
        
        data_counter_pool[data_counter_name] = data_counter_pymodule.data_counter
        data_counter_dataset_name_list_dict[data_counter_name] = data_counter_pymodule.dataset_name_list


    #
    # Prepare data counters
    #


    if not exp_data["status"]["requirements"]["data_counters_ready"]:

        with OnlyRankZero(rank, logger) as only_rank_zero:
            if only_rank_zero.is_rank_zero():

                logger.info("Prepare data counters")

                # Prepare data counter data directory

                data_counter_data_dirname = os.path.join(exp_results_data_dirname, "data_counter_data")
                if os.path.exists(data_counter_data_dirname): shutil.rmtree(data_counter_data_dirname)
                os.mkdir(data_counter_data_dirname)

                # Compute dataset to data counter associations

                dataset_name_to_data_counter_name_set_dict = {}

                for data_counter_name, data_counter_dataset_name_list in data_counter_dataset_name_list_dict.items():
                    for dataset_name in data_counter_dataset_name_list:

                        if dataset_name not in dataset_name_to_data_counter_name_set_dict:
                            dataset_name_to_data_counter_name_set_dict[dataset_name] = set()
                        dataset_name_to_data_counter_name_set_dict[dataset_name].add(data_counter_name)

                # Iterate datasets and fill up data counters

                for dataset_name, data_counter_name_set in dataset_name_to_data_counter_name_set_dict.items():

                    logger.info("  Iterate through {:s} dataset".format(dataset_name))

                    for dataset_idx in dataset_pool[dataset_name].get_split_idxs("train"):
                        dataset_metadata_point = dataset_pool[dataset_name].getitem_metadata(dataset_idx)

                        for data_counter_name in data_counter_name_set:
                            data_counter_pool[data_counter_name].count(dataset_metadata_point)
            
                # Store data counter counts

                for data_counter_name, data_counter in data_counter_pool.items():

                    data_counter_data_subdirname = os.path.join(data_counter_data_dirname, data_counter_name)
                    os.mkdir(data_counter_data_subdirname)
                    data_counter.save(data_counter_data_subdirname)

                # Update experiment data

                logger.info("Update experiment data")
                exp_data["status"]["requirements"]["data_counters_ready"] = True
                goripy.file.json.save_json(exp_data, exp_data_filename)    

        # Reload experiment data

        logger.info("Reload experiment data")
        exp_data = goripy.file.json.load_json(exp_data_filename) 


    #
    # Load data counter data
    #


    logger.info("Load data counter data")

    data_counter_data_dirname = os.path.join(exp_results_data_dirname, "data_counter_data")
    
    for data_counter_name in data_counter_name_list:

        data_counter = data_counter_pool[data_counter_name]

        data_counter_data_subdirname = os.path.join(data_counter_data_dirname, data_counter_name)
        data_counter.load(data_counter_data_subdirname)


    #
    # Build data transforms
    #


    logger.info("Build data transforms")

    data_transform_pool = {"train": {}, "eval": {}}

    for dataset_name in dataset_name_list:

        data_transform_pymodule_name = "data_transforms.{:s}".format(dataset_name)
        data_transform_pymodule = importlib.import_module(data_transform_pymodule_name)
        
        data_transform_pool["train"][dataset_name] = data_transform_pymodule.TrainDataTransform(logger)
        data_transform_pool["eval"][dataset_name] = data_transform_pymodule.EvalDataTransform(logger)


    #
    # Build modules
    #


    logger.info("Build modules")

    module_pool = {}
    module_settings_dict = {}

    module_pymodules_dirname = os.path.join(exp_results_pymodules_dirname, "modules")
    module_name_list = [filename.split(".")[0] for filename in os.listdir(module_pymodules_dirname)]
    if "__init__" in module_name_list: module_name_list.remove("__init__")
    if "__pycache__" in module_name_list: module_name_list.remove("__pycache__")

    for module_name in module_name_list:

        module_pymodule_name = "modules.{:s}".format(module_name)
        module_pymodule = importlib.import_module(module_pymodule_name)
        
        module_pool[module_name] = module_pymodule.module
        module_pool[module_name] = module_pool[module_name].to(device)

        module_settings_dict[module_name] = module_pymodule.settings


    #
    # Prepare module weights
    #


    if not exp_data["status"]["requirements"]["module_ckps_ready"]:

        with OnlyRankZero(rank, logger) as only_rank_zero:
            if only_rank_zero.is_rank_zero():

                logger.info("Prepare module weights")

                # Prepare module checkpoint directories

                module_ckp_data_dirname = os.path.join(exp_results_data_dirname, "module_ckps")
                if os.path.exists(module_ckp_data_dirname): shutil.rmtree(module_ckp_data_dirname)
                os.mkdir(module_ckp_data_dirname)

                frzn_module_ckp_data_dirname = os.path.join(module_ckp_data_dirname, "frzn")
                os.mkdir(frzn_module_ckp_data_dirname)

                init_module_ckp_data_dirname = os.path.join(module_ckp_data_dirname, "init")
                os.mkdir(init_module_ckp_data_dirname)

                last_module_ckp_data_dirname = os.path.join(module_ckp_data_dirname, "last")
                os.mkdir(last_module_ckp_data_dirname)

                module_ckp_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "module_ckps")
                
                if os.path.exists(module_ckp_temp_data_dirname): shutil.rmtree(module_ckp_temp_data_dirname)
                os.mkdir(module_ckp_temp_data_dirname)

                last_module_ckp_temp_data_dirname = os.path.join(module_ckp_temp_data_dirname, "last")
                os.mkdir(last_module_ckp_temp_data_dirname)

                # Create modules and load / save checkpoints

                module_frozen_dict = {}

                for module_name in module_name_list:

                    logger.info("  Load weights from {:s} module".format(module_name))

                    module = module_pool[module_name]
                    module_settings = module_settings_dict[module_name]
                    
                    ## Load checkpoint

                    load_weights_from_ckp = "load_weights_from_ckp" in module_settings
                    load_weights_from_exp = "load_weights_from_exp" in module_settings

                    num_load_weights_flags = sum((load_weights_from_ckp, load_weights_from_exp))

                    if num_load_weights_flags == 0:

                        logger.info("    Using default weights")

                    if num_load_weights_flags > 1:

                        logger.warn("    Found multiple weight loading options for module {:s}".format(module_name))

                    if load_weights_from_ckp:

                        orig_ckp_filename = module_settings["load_weights_from_ckp"]["ckp_filename"]

                        logger.info("    Using {:s} checkpoint".format(orig_ckp_filename))

                        orig_module_ckp_filename = os.path.join(os.environ["GORIDEEPTRAIN_DATA_HOME"], "module_ckps", orig_ckp_filename)
                        module.load_state_dict(torch.load(orig_module_ckp_filename, map_location=device))

                    if load_weights_from_exp:

                        orig_exp_name = module_settings["load_weights_from_exp"]["exp_name"]
                        orig_exp_results_dirname = os.path.join(os.environ["GORIDEEPTRAIN_DATA_HOME"], "experiment_results", orig_exp_name)

                        orig_ckp_pool_name = module_settings["load_weights_from_exp"]["ckp_pool_name"]
                        orig_module_name = module_settings["load_weights_from_exp"]["module_name"]

                        logger.info("    Using {:s} checkpoint from experiment {:s}".format(os.path.join(orig_ckp_pool_name, orig_module_name), orig_exp_name))

                        orig_module_ckp_filename = os.path.join(orig_exp_results_dirname, "data", "module_ckps", orig_ckp_pool_name, orig_module_name + ".pt")
                        module.load_state_dict(torch.load(orig_module_ckp_filename, map_location=device))

                    ## If module is frozen, module does not require grads

                    if module_settings.get("frozen", False):
                        for param in module.parameters():
                            param.requires_grad = False

                    ## Save initial checkpoint

                    module_frozen = all(not param.requires_grad for param in module.parameters())
                    module_frozen_dict[module_name] = module_frozen

                    if module_frozen:

                        ckp_filename = os.path.join(frzn_module_ckp_data_dirname, "{:s}.pt".format(module_name))
                        torch.save(module.state_dict(), ckp_filename)

                    else:

                        ckp_filename = os.path.join(init_module_ckp_data_dirname, "{:s}.pt".format(module_name))
                        torch.save(module.state_dict(), ckp_filename)

                        ckp_filename = os.path.join(last_module_ckp_data_dirname, "{:s}.pt".format(module_name))
                        torch.save(module.state_dict(), ckp_filename)

                # Save module frozen data

                module_frozen_dict_filename = os.path.join(module_ckp_data_dirname, "module_frozen_dict.json")
                goripy.file.json.save_json(module_frozen_dict, module_frozen_dict_filename)

                # Update experiment data

                logger.info("Update experiment data")
                exp_data["status"]["requirements"]["module_ckps_ready"] = True
                goripy.file.json.save_json(exp_data, exp_data_filename)    

        # Reload experiment data

        logger.info("Reload experiment data")
        exp_data = goripy.file.json.load_json(exp_data_filename) 


    #
    # Load module weights and wrap modules with DDP
    #


    logger.info("Load module weights and wrap modules with DDP")

    module_ckp_data_dirname = os.path.join(exp_results_data_dirname, "module_ckps")

    module_frozen_dict_filename = os.path.join(module_ckp_data_dirname, "module_frozen_dict.json")
    module_frozen_dict = goripy.file.json.load_json(module_frozen_dict_filename)

    for module_name in module_name_list:

        module = module_pool[module_name]

        # Load module weights

        module_ckp_pool_name = "frzn" if module_frozen_dict[module_name] else "last"

        module_ckp_filename = os.path.join(module_ckp_data_dirname, module_ckp_pool_name, "{:s}.pt".format(module_name))
        module.load_state_dict(torch.load(module_ckp_filename, map_location=device))

        ## If module is frozen, module does not require grads or DDP wrap

        if module_frozen_dict[module_name]:

            for param in module.parameters():
                param.requires_grad = False

        else:

            module = torch.nn.parallel.DistributedDataParallel(module)

        module_pool[module_name] = module


    #
    # Build optimizers and schedulers
    #


    logger.info("Build optimizers and schedulers")

    optimizer_pool = {}
    scheduler_pool = {}
    scheduler_start_epoch_dict = {}

    optim_sched_pymodules_dirname = os.path.join(exp_results_pymodules_dirname, "optim_sched")
    optim_sched_name_list = [filename.split(".")[0] for filename in os.listdir(optim_sched_pymodules_dirname)]
    if "__init__" in optim_sched_name_list: optim_sched_name_list.remove("__init__")
    if "__pycache__" in optim_sched_name_list: optim_sched_name_list.remove("__pycache__")

    for optim_sched_name in optim_sched_name_list:

        optim_sched_pymodule_name = "optim_sched.{:s}".format(optim_sched_name)
        optim_sched_pymodule = importlib.import_module(optim_sched_pymodule_name)

        optimizer = optim_sched_pymodule.create_optimizer(module_pool)
        scheduler = optim_sched_pymodule.create_scheduler(optimizer)

        optimizer_pool[optim_sched_name] = optimizer
        scheduler_pool[optim_sched_name] = scheduler

        scheduler_start_epoch_dict[optim_sched_name] = optim_sched_pymodule.scheduler_start_epoch


    #
    # Prepare optimizer and scheduler data
    #


    if not exp_data["status"]["requirements"]["optim_sched_data_ready"]:

        with OnlyRankZero(rank, logger) as only_rank_zero:
            if only_rank_zero.is_rank_zero():

                logger.info("Prepare optimizer and scheduler data")

                # Prepare data directories

                optimizer_ckp_data_dirname = os.path.join(exp_results_data_dirname, "optim_ckps")
                if os.path.exists(optimizer_ckp_data_dirname): shutil.rmtree(optimizer_ckp_data_dirname)
                os.mkdir(optimizer_ckp_data_dirname)
                
                scheduler_data_dirname = os.path.join(exp_results_data_dirname, "sched_data")
                if os.path.exists(scheduler_data_dirname): shutil.rmtree(scheduler_data_dirname)
                os.mkdir(scheduler_data_dirname)
                
                scheduler_lr_data_dirname = os.path.join(exp_results_data_dirname, "lr_data")
                if os.path.exists(scheduler_lr_data_dirname): shutil.rmtree(scheduler_lr_data_dirname)
                os.mkdir(scheduler_lr_data_dirname)

                optimizer_ckp_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "optim_ckps")
                if os.path.exists(optimizer_ckp_temp_data_dirname): shutil.rmtree(optimizer_ckp_temp_data_dirname)
                os.mkdir(optimizer_ckp_temp_data_dirname)
                
                scheduler_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "sched_data")
                if os.path.exists(scheduler_temp_data_dirname): shutil.rmtree(scheduler_temp_data_dirname)
                os.mkdir(scheduler_temp_data_dirname)

                # Save optimizer checkpoints and scheduler data

                for optim_sched_name in optim_sched_name_list:

                    logger.info("  Saving checkpoint and data for {:s} optimizer and scheduler".format(optim_sched_name))

                    optimizer = optimizer_pool[optim_sched_name]
                    scheduler = scheduler_pool[optim_sched_name]

                    optimizer_ckp_filename = os.path.join(optimizer_ckp_data_dirname, "{:s}.ckp".format(optim_sched_name))
                    torch.save(optimizer.state_dict(), optimizer_ckp_filename)

                    scheduler_data_subdirname = os.path.join(scheduler_data_dirname, optim_sched_name)
                    os.mkdir(scheduler_data_subdirname)
                    scheduler.save(scheduler_data_subdirname)

                    scheduler_lr_data_subdirname = os.path.join(scheduler_lr_data_dirname, optim_sched_name)
                    os.mkdir(scheduler_lr_data_subdirname)

                    scheduler_temp_data_subdirname = os.path.join(scheduler_temp_data_dirname, optim_sched_name)
                    os.mkdir(scheduler_temp_data_subdirname)

                # Update experiment data

                logger.info("Update experiment data")
                exp_data["status"]["requirements"]["optim_sched_data_ready"] = True
                goripy.file.json.save_json(exp_data, exp_data_filename)    

        # Reload experiment data

        logger.info("Reload experiment data")
        exp_data = goripy.file.json.load_json(exp_data_filename) 


    #
    # Load optimizer and scheduler data
    #


    logger.info("Load optimizer and scheduler data")

    optimizer_ckp_data_dirname = os.path.join(exp_results_data_dirname, "optim_ckps")
    scheduler_data_dirname = os.path.join(exp_results_data_dirname, "sched_data")

    for optim_sched_name in optim_sched_name_list:

        optimizer = optimizer_pool[optim_sched_name]
        scheduler = scheduler_pool[optim_sched_name]

        optimizer_ckp_filename = os.path.join(optimizer_ckp_data_dirname, "{:s}.ckp".format(optim_sched_name))
        optimizer.load_state_dict(torch.load(optimizer_ckp_filename, map_location=device))

        scheduler_data_subdirname = os.path.join(scheduler_data_dirname, optim_sched_name)
        scheduler.load(scheduler_data_subdirname)
    
        scheduler.initialize(scheduler_start_epoch_dict[optim_sched_name])


    #
    # Build module transforms
    #


    logger.info("Build module transforms")

    module_transform_pool = {}
    module_transform_loss_ten_reg_key_list_dict = {}

    for dataset_name in dataset_name_list:

        module_transform_pymodule_name = "module_transforms.{:s}".format(dataset_name) 
        module_transform_pymodule = importlib.import_module(module_transform_pymodule_name)

        module_transform_pool[dataset_name] = module_transform_pymodule.ModuleTransform(
            data_counter_pool,
            device,
            logger
        )

        module_transform_loss_ten_reg_key_list_dict[dataset_name] = module_transform_pymodule.loss_ten_reg_key_list
        

    #
    # Build loss registers
    #


    logger.info("Build loss registers")

    loss_reg_pool = {"train": {}, "val": {}}

    loss_reg_settings_filename = os.path.join(exp_results_settings_dirname, "loss_registers.json")
    loss_reg_settings = goripy.file.json.load_json(loss_reg_settings_filename)

    for loss_reg_key in loss_reg_settings["loss_reg_key_list"]:
        loss_reg_pool["train"][loss_reg_key] = StepWiseLossRegister()
        loss_reg_pool["val"][loss_reg_key] = EpochWiseLossRegister()


    #
    # Prepare loss register data
    #


    if not exp_data["status"]["requirements"]["loss_reg_data_ready"]:

        with OnlyRankZero(rank, logger) as only_rank_zero:
            if only_rank_zero.is_rank_zero():

                logger.info("Prepare loss register data")

                # Prepare data directories

                loss_reg_data_dirname = os.path.join(exp_results_data_dirname, "loss_reg_data")
                if os.path.exists(loss_reg_data_dirname): shutil.rmtree(loss_reg_data_dirname)
                os.mkdir(loss_reg_data_dirname)

                loss_reg_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "loss_reg_data")
                if os.path.exists(loss_reg_temp_data_dirname): shutil.rmtree(loss_reg_temp_data_dirname)
                os.mkdir(loss_reg_temp_data_dirname)

                for split_str in ["train", "val"]:

                    loss_reg_data_split_dirname = os.path.join(loss_reg_data_dirname, split_str)
                    os.mkdir(loss_reg_data_split_dirname)

                    loss_reg_temp_data_split_dirname = os.path.join(loss_reg_temp_data_dirname, split_str)
                    os.mkdir(loss_reg_temp_data_split_dirname)

                    # Save loss register data

                    for loss_reg_key in loss_reg_settings["loss_reg_key_list"]:

                        logger.info("  Saving empty data for {:s} {:s} loss register".format(split_str, loss_reg_key))

                        loss_reg = loss_reg_pool[split_str][loss_reg_key]

                        loss_reg_data_subdirname = os.path.join(loss_reg_data_split_dirname, loss_reg_key)
                        os.mkdir(loss_reg_data_subdirname)

                        loss_reg_temp_data_subdirname = os.path.join(loss_reg_temp_data_split_dirname, loss_reg_key)
                        os.mkdir(loss_reg_temp_data_subdirname)

                        loss_reg_epoch_data_filename = os.path.join(loss_reg_data_subdirname, "epoch_data.npz")
                        loss_reg.save_epoch_data(loss_reg_epoch_data_filename)

                # Update experiment data

                logger.info("Update experiment data")
                exp_data["status"]["requirements"]["loss_reg_data_ready"] = True
                goripy.file.json.save_json(exp_data, exp_data_filename)    

        # Reload experiment data

        logger.info("Reload experiment data")
        exp_data = goripy.file.json.load_json(exp_data_filename) 


    #
    # Load loss register data
    #


    logger.info("Load loss register data")

    loss_reg_data_dirname = os.path.join(exp_results_data_dirname, "loss_reg_data")

    for split_str in ["train", "val"]:

        for loss_reg_key in loss_reg_settings["loss_reg_key_list"]:

            loss_reg = loss_reg_pool[split_str][loss_reg_key]

            loss_reg_data_split_dirname = os.path.join(loss_reg_data_dirname, split_str)
            loss_reg_data_subdirname = os.path.join(loss_reg_data_split_dirname, loss_reg_key)
            loss_reg_epoch_data_filename = os.path.join(loss_reg_data_subdirname, "epoch_data.npz")
            loss_reg.load_epoch_data(loss_reg_epoch_data_filename)


    #
    # Build loss weighter
    #


    logger.info("Build loss weighter")

    loss_weighter_pymodule_name = "loss_weighter"
    loss_weighter_pymodule = importlib.import_module(loss_weighter_pymodule_name)

    loss_weighter = loss_weighter_pymodule.loss_weighter


    #
    # Prepare loss weighter data
    #


    if not exp_data["status"]["requirements"]["loss_weighter_data_ready"]:

        with OnlyRankZero(rank, logger) as only_rank_zero:
            if only_rank_zero.is_rank_zero():

                logger.info("Prepare loss weighter data")

                # Prepare data directories

                loss_weighter_data_dirname = os.path.join(exp_results_data_dirname, "loss_weighter_data")
                if os.path.exists(loss_weighter_data_dirname): shutil.rmtree(loss_weighter_data_dirname)
                os.mkdir(loss_weighter_data_dirname)

                loss_weighter_temp_data_dirname = os.path.join(exp_results_data_dirname, "loss_weighter_data")
                if os.path.exists(loss_weighter_temp_data_dirname): shutil.rmtree(loss_weighter_temp_data_dirname)
                os.mkdir(loss_weighter_temp_data_dirname)

                loss_weight_data_dirname = os.path.join(exp_results_data_dirname, "loss_weight_data")
                if os.path.exists(loss_weight_data_dirname): shutil.rmtree(loss_weight_data_dirname)
                os.mkdir(loss_weight_data_dirname)

                # Save loss weighter data

                logger.info("Saving empty data for loss weighter")

                loss_weighter.save(loss_weighter_data_dirname)

                # Update experiment data

                logger.info("Update experiment data")
                exp_data["status"]["requirements"]["loss_weighter_data_ready"] = True
                goripy.file.json.save_json(exp_data, exp_data_filename)    

        # Reload experiment data

        logger.info("Reload experiment data")
        exp_data = goripy.file.json.load_json(exp_data_filename) 


    #
    # Load loss weighter data
    #


    logger.info("Load loss weighter data")

    loss_weighter_data_dirname = os.path.join(exp_results_data_dirname, "loss_weighter_data")
    loss_weighter.load(loss_weighter_data_dirname)


    #
    # Build early stopper
    #


    logger.info("Build early stopper")

    early_stopper_pymodule_name = "early_stopper"
    early_stopper_pymodule = importlib.import_module(early_stopper_pymodule_name)

    early_stopper = early_stopper_pymodule.early_stopper


    #
    # Prepare early stopper data
    #


    if not exp_data["status"]["requirements"]["early_stopper_data_ready"]:

        with OnlyRankZero(rank, logger) as only_rank_zero:
            if only_rank_zero.is_rank_zero():

                logger.info("Prepare early stopper data")

                # Prepare data directories

                early_stopper_data_dirname = os.path.join(exp_results_data_dirname, "early_stopper_data")
                if os.path.exists(early_stopper_data_dirname): shutil.rmtree(early_stopper_data_dirname)
                os.mkdir(early_stopper_data_dirname)

                early_stopper_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "early_stopper_data")
                if os.path.exists(early_stopper_temp_data_dirname): shutil.rmtree(early_stopper_temp_data_dirname)
                os.mkdir(early_stopper_temp_data_dirname)

                # Save early stopper data

                logger.info("  Saving empty data for early stopper")

                early_stopper.save(early_stopper_data_dirname)

                # Update experiment data

                logger.info("Update experiment data")
                exp_data["status"]["requirements"]["early_stopper_data_ready"] = True
                goripy.file.json.save_json(exp_data, exp_data_filename)    

        # Reload experiment data

        logger.info("Reload experiment data")
        exp_data = goripy.file.json.load_json(exp_data_filename) 


    #
    # Load early stopper data
    #


    logger.info("Load early stopper data")

    early_stopper_data_dirname = os.path.join(exp_results_data_dirname, "early_stopper_data")
    early_stopper.load(early_stopper_data_dirname)


    #
    # Build checkpoint saver
    #


    logger.info("Build checkpoint saver")

    checkpoint_saver_pymodule_name = "checkpoint_saver"
    checkpoint_saver_pymodule = importlib.import_module(checkpoint_saver_pymodule_name)

    checkpoint_saver = checkpoint_saver_pymodule.checkpoint_saver


    #
    # Prepare checkpoint saver data
    #


    if not exp_data["status"]["requirements"]["checkpoint_saver_data_ready"]:

        with OnlyRankZero(rank, logger) as only_rank_zero:
            if only_rank_zero.is_rank_zero():

                logger.info("Prepare checkpoint saver data")

                # Prepare data directories

                checkpoint_saver_data_dirname = os.path.join(exp_results_data_dirname, "checkpoint_saver_data")
                if os.path.exists(checkpoint_saver_data_dirname): shutil.rmtree(checkpoint_saver_data_dirname)
                os.mkdir(checkpoint_saver_data_dirname)

                checkpoint_saver_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "checkpoint_saver_data")
                if os.path.exists(checkpoint_saver_temp_data_dirname): shutil.rmtree(checkpoint_saver_temp_data_dirname)
                os.mkdir(checkpoint_saver_temp_data_dirname)

                # Save checkpoint saver data

                logger.info("  Saving empty data for checkpoint saver")

                checkpoint_saver.save(checkpoint_saver_data_dirname)

                # Update experiment data

                logger.info("Update experiment data")
                exp_data["status"]["requirements"]["checkpoint_saver_data_ready"] = True
                goripy.file.json.save_json(exp_data, exp_data_filename)    

        # Reload experiment data

        logger.info("Reload experiment data")
        exp_data = goripy.file.json.load_json(exp_data_filename) 


    #
    # Load checkpoint saver data
    #


    logger.info("Load checkpoint saver data")

    checkpoint_saver_data_dirname = os.path.join(exp_results_data_dirname, "checkpoint_saver_data")
    checkpoint_saver.load(checkpoint_saver_data_dirname)



    # After reaching this code line, we have: TODO
    #
    # dataset_pool
    # data_counter_pool
    # data_transform_pool
    #   train
    #   eval
    # module_pool
    # module_is_ddpd_pool
    # optimizer_pool
    # scheduler_pool
    # module_transform_pool
    # module_transform_loss_ten_reg_key_list_dict
    # loss_reg_pool
    #   train
    #   val
    # loss_weighter
    # early_stopper


    #
    # Main loop
    #


    logger.info("Begin main loop")

    while exp_data["status"]["pipeline_state"]["stage_name"] != "finished":

        logger.info("Pipeline state:")
        logger.info("  epoch {:d}".format(exp_data["status"]["pipeline_state"]["epoch_num"]))
        logger.info("  stage {:s}".format(exp_data["status"]["pipeline_state"]["stage_name"]))


        #
        # Stage: epoch_0_evaluation_loop
        #


        if exp_data["status"]["pipeline_state"]["stage_name"] == "epoch_0_evaluation_loop":

            # Run evaluation loop

            logger.info("Run evaluation loop")

            evaluation_loop(
                command_args,
                exp_data,

                data_loading_settings,
                logging_settings,

                dataset_pool,
                dataset_name_list,
                data_transform_pool,
                module_pool,
                module_transform_pool,
                module_transform_loss_ten_reg_key_list_dict,
                loss_reg_pool,
                loss_weighter,

                logger,
                tqdm_logger
            )

            # After evaluation loop

            with OnlyRankZero(rank, logger) as only_rank_zero:
                if only_rank_zero.is_rank_zero():

                    logger.info("After evaluation loop")

                    temp_data_cleanup_list = []

                    ## Uodate early stopper and checkpoint saver

                    early_stopper.update(
                        loss_reg_pool,
                        loss_weighter
                    )

                    checkpoint_saver.update(
                        loss_reg_pool,
                        loss_weighter,
                        early_stopper
                    )

                    ## Save scheduler data

                    logger.info("Save scheduler data")

                    for optim_sched_name in optim_sched_name_list:

                        scheduler = scheduler_pool[optim_sched_name]

                        scheduler_data_subdirname = os.path.join(exp_results_temp_data_dirname, "sched_data", optim_sched_name)
                        scheduler.save(scheduler_data_subdirname)

                    temp_data_cleanup_list.append("sched_data")

                    ## Save loss register data

                    logger.info("Save loss register data")

                    for loss_reg_key in loss_reg_settings["loss_reg_key_list"]:
                        
                        loss_reg = loss_reg_pool["val"][loss_reg_key]

                        loss_reg_epoch_data_filename = os.path.join("loss_reg_data", "val", loss_reg_key, "epoch_data.npz")

                        loss_reg_epoch_data_full_filename = os.path.join(exp_results_temp_data_dirname, loss_reg_epoch_data_filename)

                        loss_reg.save_epoch_data(loss_reg_epoch_data_full_filename)

                        temp_data_cleanup_list.append(loss_reg_epoch_data_filename)

                    ## Save loss weighter data

                    logger.info("Save loss weighter data")

                    loss_weighter_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "loss_weighter_data")
                    loss_weighter.save(loss_weighter_temp_data_dirname)

                    loss_weight_data_dirname = os.path.join(exp_results_data_dirname, "loss_weight_data")
                    loss_weight_data_filename = os.path.join(loss_weight_data_dirname, "loss_weights_{:06d}.json".format(exp_data["status"]["pipeline_state"]["epoch_num"]))
                    loss_weighter.save_loss_weights(loss_weight_data_filename, loss_reg_settings["loss_reg_key_list"])

                    temp_data_cleanup_list.append("loss_weighter_data")

                    ## Save early stopper data

                    logger.info("Save early stopper data")
                    
                    early_stopper_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "early_stopper_data")
                    early_stopper.save(early_stopper_temp_data_dirname)

                    temp_data_cleanup_list.append("early_stopper_data")

                    ## Save checkpoint saver data

                    logger.info("Save checkpoint saver data")
                    
                    checkpoint_saver_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "checkpoint_saver_data")
                    checkpoint_saver.save(checkpoint_saver_temp_data_dirname)

                    temp_data_cleanup_list.append("checkpoint_saver_data")

                    ## Update experiment data

                    logger.info("Update experiment data")
                    exp_data["status"]["pipeline_state"]["epoch_num"] += 1
                    exp_data["status"]["pipeline_state"]["stage_name"] = "training_loop"
                    exp_data["status"]["pipeline_state"]["temp_data_cleanup_list"] = temp_data_cleanup_list
                    goripy.file.json.save_json(exp_data, exp_data_filename)

            # Reload experiment data

            logger.info("Reload experiment data")
            exp_data = goripy.file.json.load_json(exp_data_filename) 


        #
        # Stage: training_loop
        #


        elif exp_data["status"]["pipeline_state"]["stage_name"] == "training_loop":

            # Run training loop

            logger.info("Run training loop")

            training_loop(
                command_args,
                exp_data,

                data_loading_settings,
                logging_settings,

                dataset_pool,
                dataset_name_list,
                data_transform_pool,
                module_pool,
                optimizer_pool,
                scheduler_pool,
                module_transform_pool,
                module_transform_loss_ten_reg_key_list_dict,
                loss_reg_pool,
                loss_weighter,

                logger,
                tqdm_logger,
            )

            # After training loop

            with OnlyRankZero(rank, logger) as only_rank_zero:
                if only_rank_zero.is_rank_zero():

                    logger.info("After training loop")

                    temp_data_cleanup_list = []

                    ## Save model checkpoints

                    logger.info("Save model checkpoints")

                    for module_name in module_name_list:

                        module = module_pool[module_name]

                        if not module_frozen_dict[module_name]:

                            ckp_filename = os.path.join(exp_results_temp_data_dirname, "module_ckps", "last", "{:s}.pt".format(module_name))
                            torch.save(module.module.state_dict(), ckp_filename)

                    temp_data_cleanup_list.append(os.path.join("module_ckps", "last"))

                    ## Save optimizer checkpoint and scheduler data

                    logger.info("Save optimizer checkpoint and scheduler data")

                    for optim_sched_name in optim_sched_name_list:

                        optimizer = optimizer_pool[optim_sched_name]
                        scheduler = scheduler_pool[optim_sched_name]

                        optimizer_ckp_filename = os.path.join(exp_results_temp_data_dirname, "optim_ckps", "{:s}.ckp".format(optim_sched_name))
                        torch.save(optimizer.state_dict(), optimizer_ckp_filename)

                        scheduler_data_subdirname = os.path.join(exp_results_temp_data_dirname, "sched_data", optim_sched_name)
                        scheduler.save(scheduler_data_subdirname)

                        scheduler_epoch_lr_data_dirname = os.path.join(exp_results_data_dirname, "lr_data", optim_sched_name, "epoch_{:06d}".format(exp_data["status"]["pipeline_state"]["epoch_num"]))
                        if not os.path.exists(scheduler_epoch_lr_data_dirname): os.mkdir(scheduler_epoch_lr_data_dirname)
                        scheduler.save_epoch_lr_data(scheduler_epoch_lr_data_dirname)

                    temp_data_cleanup_list.append("optim_ckps")
                    temp_data_cleanup_list.append("sched_data")

                    ## Save loss register data

                    logger.info("Save loss register data")

                    for loss_reg_key in loss_reg_settings["loss_reg_key_list"]:
                        
                        loss_reg = loss_reg_pool["train"][loss_reg_key]

                        loss_reg_step_data_filename = os.path.join("loss_reg_data", "train", loss_reg_key, "step_data_{:06d}.npz".format(exp_data["status"]["pipeline_state"]["epoch_num"]))
                        loss_reg_epoch_data_filename = os.path.join("loss_reg_data", "train", loss_reg_key, "epoch_data.npz")

                        loss_reg_step_data_full_filename = os.path.join(exp_results_data_dirname, loss_reg_step_data_filename)
                        loss_reg_epoch_data_full_filename = os.path.join(exp_results_temp_data_dirname, loss_reg_epoch_data_filename)

                        loss_reg.save_step_data(loss_reg_step_data_full_filename)
                        loss_reg.save_epoch_data(loss_reg_epoch_data_full_filename)

                        temp_data_cleanup_list.append(loss_reg_epoch_data_filename)

                    ## Save loss weighter data

                    logger.info("Save loss weighter data")

                    loss_weighter_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "loss_weighter_data")
                    loss_weighter.save(loss_weighter_temp_data_dirname)

                    temp_data_cleanup_list.append("loss_weighter_data")

                    ## Update experiment data

                    logger.info("Update experiment data")
                    exp_data["status"]["pipeline_state"]["stage_name"] = "evaluation_loop"
                    exp_data["status"]["pipeline_state"]["temp_data_cleanup_list"] = temp_data_cleanup_list
                    goripy.file.json.save_json(exp_data, exp_data_filename)

            # Reload experiment data

            logger.info("Reload experiment data")
            exp_data = goripy.file.json.load_json(exp_data_filename)


        #
        # Stage: evaluation_loop
        #

        
        elif exp_data["status"]["pipeline_state"]["stage_name"] == "evaluation_loop":

            # Run evaluation loop

            logger.info("Run evaluation loop")

            evaluation_loop(
                command_args,
                exp_data,

                data_loading_settings,
                logging_settings,

                dataset_pool,
                dataset_name_list,
                data_transform_pool,
                module_pool,
                module_transform_pool,
                module_transform_loss_ten_reg_key_list_dict,
                loss_reg_pool,
                loss_weighter,

                logger,
                tqdm_logger,
            )

            # After evaluation loop

            with OnlyRankZero(rank, logger) as only_rank_zero:
                if only_rank_zero.is_rank_zero():

                    logger.info("After evaluation loop")

                    temp_data_cleanup_list = []

                    ## Uodate early stopper and checkpoint saver

                    early_stopper.update(
                        loss_reg_pool,
                        loss_weighter
                    )

                    checkpoint_saver.update(
                        loss_reg_pool,
                        loss_weighter,
                        early_stopper
                    )

                    ## Save model checkpoint copies

                    logger.info("Save model checkpoint copies")

                    module_ckp_data_dirname = os.path.join(exp_results_data_dirname, "module_ckps")
                    last_module_ckp_data_dirname = os.path.join(module_ckp_data_dirname, "last")

                    if early_stopper.improvement():

                        best_module_ckp_data_dirname = os.path.join(module_ckp_data_dirname, "best")
                        if os.path.exists(best_module_ckp_data_dirname): shutil.rmtree(best_module_ckp_data_dirname)
                        shutil.copytree(last_module_ckp_data_dirname, best_module_ckp_data_dirname)

                    if checkpoint_saver.save_checkpoints():

                        epoch_module_ckp_data_dirname = os.path.join(module_ckp_data_dirname, "epoch_{:06d}".format(exp_data["status"]["pipeline_state"]["epoch_num"]))
                        if os.path.exists(epoch_module_ckp_data_dirname): shutil.rmtree(epoch_module_ckp_data_dirname)
                        shutil.copytree(last_module_ckp_data_dirname, epoch_module_ckp_data_dirname)

                    ## Save scheduler data

                    logger.info("Save scheduler data")

                    for optim_sched_name in optim_sched_name_list:

                        scheduler = scheduler_pool[optim_sched_name]

                        scheduler_data_subdirname = os.path.join(exp_results_temp_data_dirname, "sched_data", optim_sched_name)
                        scheduler.save(scheduler_data_subdirname)

                    temp_data_cleanup_list.append("sched_data")

                    ## Save loss register data

                    logger.info("Save loss register data")

                    for loss_reg_key in loss_reg_settings["loss_reg_key_list"]:
                        
                        loss_reg = loss_reg_pool["val"][loss_reg_key]

                        loss_reg_epoch_data_filename = os.path.join("loss_reg_data", "val", loss_reg_key, "epoch_data.npz")

                        loss_reg_epoch_data_full_filename = os.path.join(exp_results_temp_data_dirname, loss_reg_epoch_data_filename)

                        loss_reg.save_epoch_data(loss_reg_epoch_data_full_filename)

                        temp_data_cleanup_list.append(loss_reg_epoch_data_filename)

                    ## Save loss weighter data

                    logger.info("Save loss weighter data")

                    loss_weighter_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "loss_weighter_data")
                    loss_weighter.save(loss_weighter_temp_data_dirname)

                    loss_weight_data_dirname = os.path.join(exp_results_data_dirname, "loss_weight_data")
                    loss_weight_data_filename = os.path.join(loss_weight_data_dirname, "loss_weights_{:06d}.json".format(exp_data["status"]["pipeline_state"]["epoch_num"]))
                    loss_weighter.save_loss_weights(loss_weight_data_filename, loss_reg_settings["loss_reg_key_list"])

                    temp_data_cleanup_list.append("loss_weighter_data")

                    ## Save early stopper data

                    logger.info("Save early stopper data")
                    
                    early_stopper_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "early_stopper_data")
                    early_stopper.save(early_stopper_temp_data_dirname)

                    temp_data_cleanup_list.append("early_stopper_data")

                    ## Save checkpoint saver data

                    logger.info("Save checkpoint saver data")
                    
                    checkpoint_saver_temp_data_dirname = os.path.join(exp_results_temp_data_dirname, "checkpoint_saver_data")
                    checkpoint_saver.save(checkpoint_saver_temp_data_dirname)

                    temp_data_cleanup_list.append("checkpoint_saver_data")

                    ## Update experiment data

                    logger.info("Update experiment data")

                    if early_stopper.early_stop():

                        exp_data["status"]["pipeline_state"]["stage_name"] = "finished"

                    else:

                        exp_data["status"]["pipeline_state"]["epoch_num"] += 1
                        exp_data["status"]["pipeline_state"]["stage_name"] = "training_loop"

                    exp_data["status"]["pipeline_state"]["temp_data_cleanup_list"] = temp_data_cleanup_list

                    goripy.file.json.save_json(exp_data, exp_data_filename)                    

            # Reload experiment data

            logger.info("Reload experiment data")
            exp_data = goripy.file.json.load_json(exp_data_filename)


        #
        # Cleanup (post-stage)
        #


        if len(exp_data["status"]["pipeline_state"]["temp_data_cleanup_list"]) > 0:

            with OnlyRankZero(rank, logger) as only_rank_zero:
                if only_rank_zero.is_rank_zero():

                    logger.info("Perform experiment result temp data cleanup")

                    ## Perform experiment result temp data cleanup

                    for cleanup_data_name in exp_data["status"]["pipeline_state"]["temp_data_cleanup_list"]:

                        full_data_name = os.path.join(exp_results_data_dirname, cleanup_data_name)
                        temp_full_data_name = os.path.join(exp_results_temp_data_dirname, cleanup_data_name)

                        if os.path.exists(temp_full_data_name):

                            if os.path.isfile(temp_full_data_name):

                                if os.path.exists(full_data_name): os.remove(full_data_name)
                                shutil.copyfile(temp_full_data_name, full_data_name)

                            if os.path.isdir(temp_full_data_name):

                                if os.path.exists(full_data_name): shutil.rmtree(full_data_name)
                                shutil.copytree(temp_full_data_name, full_data_name)

                            logger.info("Processed temp data item \"{:s}\"".format(
                                cleanup_data_name
                            ))

                    ## Update experiment data

                    logger.info("Update experiment data")
                    exp_data["status"]["pipeline_state"]["temp_data_cleanup_list"] = []
                    goripy.file.json.save_json(exp_data, exp_data_filename)

            # Reload experiment data

            logger.info("Reload experiment data")
            exp_data = goripy.file.json.load_json(exp_data_filename) 

    
    logger.info("Experiment finished")



############
# ENTRYPOINT
############



if __name__ == '__main__':


    #
    # GPU initilization
    #


    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend="nccl")


    #
    # Run experiment
    #


    try:

        experiment_pipeline()

    except AbortExperimentError as ex:

        if rank == 0:

            sys.stderr.write("Experiment aborted:\n\n")
            sys.stderr.write(traceback.format_exc())
            sys.stderr.write("\n")
            sys.stderr.write("Original traceback:\n\n")
            sys.stderr.write(ex.orig_traceback)

        else:

            sys.stderr.write("Experiment aborted: An error occurred in rank 0")

    except Exception:

        sys.stderr.write("An unexpected error occured in rank {:d}\n".format(rank))
        sys.stderr.write(traceback.format_exc())


    #
    # GPU finalization
    #


    torch.distributed.destroy_process_group()
