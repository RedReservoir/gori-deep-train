import os
import sys
import shutil
import argparse
import logging
import datetime
import importlib
import multiprocessing

import numpy
import torch

####

import goripy.file.json
import goripy.log
import goripy.tqdm
import goripy.array.chunk
import goripy.memory.info

####

from gorideep.utils.metadata import CategoryMetadata



#######################
# EVALUATION SUBPROCESS
#######################



def compute_cat_logits_probs_subprocess(
    rank,
    
    command_args,
    
    logging_settings,
    cat_metadata,

    dataset_name,
    split_str,

    dataset_split_idx_chunk,
    pred_cat_logits_temp_filename,
    target_cat_probs_temp_filename,
    cat_weights_temp_filename
):


    #
    # Device initialization
    #


    world_size = torch.cuda.device_count()

    device = torch.device(rank)


    #
    # Define evaluation directories
    #


    eval_settings_dirname = os.path.join(os.environ["GORIDEEPTRAIN_DATA_HOME"], "evaluation_settings", command_args.eval_name)
    eval_results_dirname = os.path.join(os.environ["GORIDEEPTRAIN_DATA_HOME"], "evaluation_results", command_args.eval_name)
    eval_logs_dirname = os.path.join(os.environ["GORIDEEPTRAIN_DATA_HOME"], "evaluation_logs", command_args.eval_name)


    #
    # Setup logging
    #


    # Create logger

    logger_name = "rank_{:d}".format(rank)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter("%(name)10s > %(asctime)s %(levelname)8s: %(message)s")

    # Create file handler

    logger_filename = os.path.join(eval_logs_dirname, "{:s}.log".format(logger_name))
    log_file_handler = logging.FileHandler(filename=logger_filename, mode="a")
    log_file_handler.setFormatter(log_formatter)

    logger.addHandler(log_file_handler)

    # Create tqdm logger

    tqdm_logger = goripy.log.TqdmLogger(
        logger,
        log_level=logging.DEBUG
    )

    # Redirect stderr

    sys.stderr = goripy.log.StderrLogger(
        logger,
        log_level=logging.ERROR
    )

    logger.info("Initialized inference subprocess - {:s} dataset - {:s} split".format(dataset_name, split_str))



    #
    # Define evaluation result directories
    #

    
    eval_results_settings_dirname = os.path.join(eval_results_dirname, "settings")    
    eval_results_pymodules_dirname = os.path.join(eval_results_dirname, "pymodules")
    eval_results_data_dirname = os.path.join(eval_results_dirname, "data")


    #
    # Build dataset
    #


    logger.info("Build dataset")

    dataset_pymodule_name = "datasets.{:s}".format(dataset_name)
    dataset_pymodule = importlib.import_module(dataset_pymodule_name)

    dataset = dataset_pymodule.dataset


    #
    # Build data transform
    #


    logger.info("Build data transform")

    data_transform_pymodule_name = "data_transforms.{:s}".format(dataset_name) 
    data_transform_pymodule = importlib.import_module(data_transform_pymodule_name)

    data_transform = data_transform_pymodule.DataTransform(logger)
    

    #
    # Build modules
    #


    logger.info("Build modules")

    module_pool = {}
    module_settings_dict = {}

    module_pymodules_dirname = os.path.join(eval_results_pymodules_dirname, "modules")
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
    # Load module weights
    #


    logger.info("Load module weights")

    for module_name in module_pool.keys():

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

            logger.warning("    Found multiple weight loading options for module {:s}".format(module_name))

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

        ## Prepare module for inference

        module = module.eval()

        for param in module.parameters():
            param.requires_grad = False

        module_pool[module_name] = module


    #
    # Build module transform
    #


    logger.info("Build module transform")

    module_transform_pymodule_name = "module_transforms.{:s}".format(dataset_name) 
    module_transform_pymodule = importlib.import_module(module_transform_pymodule_name)

    module_transform = module_transform_pymodule.ModuleTransform(
        None,
        device,
        logger
    )

    cat_key_dict = module_transform_pymodule.cat_key_dict


    #
    # Build dataloader
    #


    logger.info("Build dataloader")

    # Prepare dataset chunk
    
    dataset.set_data_transform(data_transform)
    _ = dataset[0]
    chunk_dataset = torch.utils.data.Subset(dataset, dataset_split_idx_chunk)

    # Prepare dataloader

    dataloader_pymodule_name = "dataloaders.{:s}".format(dataset_name)
    dataloader_pymodule = importlib.import_module(dataloader_pymodule_name)

    dataloader_collate_fn = dataloader_pymodule.dataloader_collate_fn

    dataloader_batch_size = dataloader_pymodule.dataloader_args["batch_size"]
    dataloader_num_workers = dataloader_pymodule.dataloader_args["num_workers"]
    dataloader_prefetch_factor = dataloader_pymodule.dataloader_args["prefetch_factor"]

    chunk_dataloader = torch.utils.data.DataLoader(
        dataset=chunk_dataset,
        batch_size=dataloader_batch_size,
        num_workers=dataloader_num_workers,
        collate_fn=dataloader_collate_fn,
        pin_memory=True,
        prefetch_factor=dataloader_prefetch_factor,
    )


    #
    # Inference loop
    #


    logger.info("Inference loop")

    # Prepare logit and prob buffers

    pred_cat_logit_tenn = torch.empty(
        size=(dataset_split_idx_chunk.shape[0], cat_metadata.get_num_cats()),
        dtype=torch.float,
        device=device
    )

    target_cat_prob_tenn = torch.empty(
        size=(dataset_split_idx_chunk.shape[0], cat_metadata.get_num_cats()),
        dtype=torch.float,
        device=device
    )

    cat_weight_tenn = torch.empty(
        size=(dataset_split_idx_chunk.shape[0], cat_metadata.get_num_cats()),
        dtype=torch.float,
        device=device
    )

    curr_data_point_idx = 0

    # Loop through dataloader

    if logging_settings["tqdm"]["enabled"]:
        chunk_dataloader = goripy.tqdm.tqdmidify(
            chunk_dataloader,
            tqdm_len=len(chunk_dataloader),
            tqdm_freq=logging_settings["tqdm"]["freq"],
            tqdm_file=tqdm_logger
        )

    for data_batch in chunk_dataloader:

        with torch.no_grad():

            ## Forward pass

            for data_batch_key, data_batch_value in data_batch.items():
                if type(data_batch_value) is torch.Tensor:
                    data_batch[data_batch_key] = data_batch_value.to(device)

            module_transform(
                data_batch,
                module_pool
            )
                    
            ## Accumulate data

            batch_size = data_batch["batch_size"]

            start_idx = curr_data_point_idx
            end_idx = start_idx + batch_size

            batch_pred_cat_logit_ten = data_batch[cat_key_dict["pred_cat_logit_ten_key"]]
            batch_target_cat_prob_ten = data_batch[cat_key_dict["target_cat_prob_ten_key"]]
            batch_cat_weight_ten = data_batch[cat_key_dict["cat_weight_ten_key"]]

            pred_cat_logit_tenn[start_idx:end_idx] = batch_pred_cat_logit_ten[:]
            target_cat_prob_tenn[start_idx:end_idx] = batch_target_cat_prob_ten[:]
            cat_weight_tenn[start_idx:end_idx] = batch_cat_weight_ten[:]

            curr_data_point_idx += batch_size


    #
    # Report GPU memory utilization
    #


    free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(device)
    used_mem_bytes = total_mem_bytes - free_mem_bytes

    used_mem_str = goripy.memory.info.sprint_fancy_num_bytes(used_mem_bytes, unit="GiB")
    total_mem_str = goripy.memory.info.sprint_fancy_num_bytes(total_mem_bytes, unit="GiB")

    logger.info("Peak GPU memory utilization: [{:s}] / [{:s}]".format(used_mem_str, total_mem_str))    


    #
    # Save temp logits and probs
    #


    logger.info("Saving predicted category logits to {:s}".format(pred_cat_logits_temp_filename))
    torch.save(pred_cat_logit_tenn, pred_cat_logits_temp_filename)

    logger.info("Saving target category probabilities to {:s}".format(target_cat_probs_temp_filename))
    torch.save(target_cat_prob_tenn, target_cat_probs_temp_filename)

    logger.info("Saving category weights to {:s}".format(cat_weights_temp_filename))
    torch.save(cat_weight_tenn, cat_weights_temp_filename)



############
# ENTRYPOINT
############



if __name__ == '__main__':


    multiprocessing.set_start_method("spawn")


    #
    # Device initialization
    #


    world_size = torch.cuda.device_count()

    device = torch.device(0)


    #
    # Capture current time
    #


    run_datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


    #
    # Parse command args
    #


    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "eval_name",
        help="name of the evaluation run"
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
    
    command_args = parser.parse_args()
    

    #
    # Create evaluation directories
    #


    eval_settings_dirname = os.path.join(os.environ["GORIDEEPTRAIN_DATA_HOME"], "evaluation_settings", command_args.eval_name)
    eval_results_dirname = os.path.join(os.environ["GORIDEEPTRAIN_DATA_HOME"], "evaluation_results", command_args.eval_name)
    eval_logs_dirname = os.path.join(os.environ["GORIDEEPTRAIN_DATA_HOME"], "evaluation_logs", command_args.eval_name)

    eval_settings_settings_dirname = os.path.join(eval_settings_dirname, "settings")
    eval_settings_pymodules_dirname = os.path.join(eval_settings_dirname, "pymodules")


    if command_args.resume and command_args.reset:

        raise ValueError("Flags --resume and --reset cannot be active simultaneously")

    elif command_args.resume:

        if not os.path.exists(eval_results_dirname):
            raise ValueError("Evaluation results directory does not exist but --resume flag is enabled")

        if not os.path.exists(eval_logs_dirname):
            raise ValueError("Evaluation logs directory does not exist but --resume flag is enabled")

    elif command_args.reset:

        if os.path.exists(eval_results_dirname): shutil.rmtree(eval_results_dirname)
        os.makedirs(eval_results_dirname, exist_ok=True)

        if os.path.exists(eval_logs_dirname): shutil.rmtree(eval_logs_dirname)
        os.makedirs(eval_logs_dirname, exist_ok=True)
    
    else:

        if os.path.exists(eval_results_dirname):
            raise ValueError("Evaluation results directory already exists, provide the --resume or the --reset flag")
        
        os.makedirs(eval_results_dirname, exist_ok=True)

        if os.path.exists(eval_results_dirname):
            raise ValueError("Evaluation logs directory already exists, provide the --resume or the --reset flag")

        os.makedirs(eval_logs_dirname, exist_ok=True)


    #
    # Setup logging
    #


    # Create logger

    logger_name = "main"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter("%(name)10s > %(asctime)s %(levelname)8s: %(message)s")

    # Create file handler

    logger_filename = os.path.join(eval_logs_dirname, "{:s}.log".format(logger_name))
    log_file_handler = logging.FileHandler(filename=logger_filename, mode="a")
    log_file_handler.setFormatter(log_formatter)

    logger.addHandler(log_file_handler)

    # Create tqdm logger

    tqdm_logger = goripy.log.TqdmLogger(
        logger,
        log_level=logging.DEBUG
    )

    # Redirect stderr

    sys.stderr = goripy.log.StderrLogger(
        logger,
        log_level=logging.ERROR
    )


    #
    # Prepare evaluation data
    #

    
    eval_data_filename = os.path.join(eval_results_dirname, "eval_data.json")
    
    # Create or update evaluation data

    if os.path.exists(eval_data_filename):
    
        logger.info("Update evaluation data")
        
        eval_data = goripy.file.json.load_json(eval_data_filename)
        eval_data["status"]["run_datetime_list"].append(run_datetime_str)
    
    else:

        logger.info("Create evaluation data")

        eval_data = {}

        eval_data["metadata"] = {}

        eval_data["status"] = {
            "run_datetime_list": [run_datetime_str],
            "requirements": {
                "eval_results_settings_ready": False,
                "eval_results_pymodules_ready": False,
                "eval_data_progress_ready": False
            }
        }

    goripy.file.json.save_json(eval_data, eval_data_filename)

    # Load evaluation data

    logger.info("Load evaluation data")
    eval_data = goripy.file.json.load_json(eval_data_filename)

    logger.info("Requirements:")
    for req_key, req_value in eval_data["status"]["requirements"].items():
        logger.info("  {:30s} {:>5s}".format(req_key, str(req_value)))


    #
    # Define evaluation result directories
    #

    
    eval_results_settings_dirname = os.path.join(eval_results_dirname, "settings")    
    eval_results_pymodules_dirname = os.path.join(eval_results_dirname, "pymodules")
    eval_results_data_dirname = os.path.join(eval_results_dirname, "data")


    #
    # Prepare evaluation settings
    #
    

    if not eval_data["status"]["requirements"]["eval_results_settings_ready"]:

        logger.info("Prepare experiment results settings")

        ## Create evaluation settings directory

        if os.path.exists(eval_results_settings_dirname): shutil.rmtree(eval_results_settings_dirname)
        os.mkdir(eval_results_settings_dirname)

        ## Copy evaluation settings
        
        for settings_filename in [
            "logging.json",
            "metadata.json"
        ]:
            
            settings_src_full_filename = os.path.join(eval_settings_settings_dirname, settings_filename)
            settings_dst_full_filename = os.path.join(eval_results_settings_dirname, settings_filename)

            if not os.path.exists(settings_src_full_filename):
                raise ValueError("Missing settings file: {:s}".format(settings_src_full_filename))

            shutil.copyfile(settings_src_full_filename, settings_dst_full_filename)

        ## Update evaluation data

        logger.info("Update experiment data")
        eval_data["status"]["requirements"]["eval_results_settings_ready"] = True
        goripy.file.json.save_json(eval_data, eval_data_filename)


    #
    # Prepare evaluation results pymodules
    #
    

    if not eval_data["status"]["requirements"]["eval_results_pymodules_ready"]:

        logger.info("Prepare evaluation pymodules")

        ## Create evaluation pymodules directory

        if os.path.exists(eval_results_pymodules_dirname): shutil.rmtree(eval_results_pymodules_dirname)
        os.mkdir(eval_results_pymodules_dirname)

        ## Copy evaluation pymodules
        
        for pymodules_dirname in [
            "data_transforms",
            "dataloaders",
            "datasets",
            "module_transforms",
            "modules"
        ]:
            
            pymodules_src_full_dirname = os.path.join(eval_settings_pymodules_dirname, pymodules_dirname)
            pymodules_dst_full_dirname = os.path.join(eval_results_pymodules_dirname, pymodules_dirname)

            if not os.path.exists(pymodules_src_full_dirname):
                raise ValueError("Missing pymodules directory: {:s}".format(settings_src_full_filename))
            
            shutil.copytree(pymodules_src_full_dirname, pymodules_dst_full_dirname)

            pymodules_init_full_filename = os.path.join(pymodules_dst_full_dirname, "__init__.py")
            open(pymodules_init_full_filename, "w").close()

        ## Update evaluation data

        logger.info("Update experiment data")
        eval_data["status"]["requirements"]["eval_results_pymodules_ready"] = True
        goripy.file.json.save_json(eval_data, eval_data_filename)

    # Add evaluation pymodules root path to sys.path

    sys.path.insert(0, eval_results_pymodules_dirname)


    #
    # Load logging settings
    #


    logger.info("Load logging settings")

    logging_settings_dirname = os.path.join(eval_results_settings_dirname, "logging.json")
    logging_settings = goripy.file.json.load_json(logging_settings_dirname)


    #
    # Load metadata settings
    #


    logger.info("Load category settings")

    metadata_settings_dirname = os.path.join(eval_results_settings_dirname, "metadata.json")
    metadata_settings = goripy.file.json.load_json(metadata_settings_dirname)

    cat_metadata = CategoryMetadata(metadata_settings["cat_subset_name"])


    #
    # Build datasets
    #


    logger.info("Build datasets")

    dataset_pool = {}

    dataset_pymodules_dirname = os.path.join(eval_results_pymodules_dirname, "datasets")
    dataset_name_list = [filename.split(".")[0] for filename in os.listdir(dataset_pymodules_dirname)]
    if "__init__" in dataset_name_list: dataset_name_list.remove("__init__")
    if "__pycache__" in dataset_name_list: dataset_name_list.remove("__pycache__")

    for dataset_name in dataset_name_list:

        dataset_pymodule_name = "datasets.{:s}".format(dataset_name)
        dataset_pymodule = importlib.import_module(dataset_pymodule_name)

        dataset_pool[dataset_name] = dataset_pymodule.dataset


    #
    # Prepare evaluation data progress
    #
 
 
    if not eval_data["status"]["requirements"]["eval_data_progress_ready"]:
        
        logger.info("Prepare evaluation data progress")

        eval_data["status"]["progress"] = {
            dataset_name: {split_str: False for split_str in ["train", "val", "test"]}
            for dataset_name in dataset_name_list
        }

        ## Update evaluation data

        logger.info("Update experiment data")
        eval_data["status"]["requirements"]["eval_data_progress_ready"] = True
        goripy.file.json.save_json(eval_data, eval_data_filename)


    #
    # Run category inference subprocesses
    #


    os.makedirs(eval_results_data_dirname, exist_ok=True)

    pred_cat_logits_dirname = os.path.join(eval_results_data_dirname, "pred_cat_logits")
    target_cat_probs_dirname = os.path.join(eval_results_data_dirname, "target_cat_probs")
    cat_weights_dirname = os.path.join(eval_results_data_dirname, "cat_weights")

    os.makedirs(pred_cat_logits_dirname, exist_ok=True)
    os.makedirs(target_cat_probs_dirname, exist_ok=True)
    os.makedirs(cat_weights_dirname, exist_ok=True)
    
    for dataset_name in dataset_name_list:

        dataset = dataset_pool[dataset_name]

        # Create dataset result directories

        pred_cat_logits_dataset_dirname = os.path.join(pred_cat_logits_dirname, dataset_name)
        target_cat_probs_dataset_dirname = os.path.join(target_cat_probs_dirname, dataset_name)
        cat_weights_dataset_dirname = os.path.join(cat_weights_dirname, dataset_name)

        os.makedirs(pred_cat_logits_dataset_dirname, exist_ok=True)
        os.makedirs(target_cat_probs_dataset_dirname, exist_ok=True)
        os.makedirs(cat_weights_dataset_dirname, exist_ok=True)

        # Process dataset splits

        for split_str in ["train", "val", "test"]:

            logger.info("Processing {:s} dataset - {:s} split".format(dataset_name, split_str))

            # If dataset and split have been processed, skip

            if eval_data["status"]["progress"][dataset_name][split_str]:

                logger.info("Skipping: Already processed")

                continue

            # If split is missing or empty, skip

            dataset_split_idxs = dataset.get_split_idxs(split_str)

            if len(dataset_split_idxs) == 0:

                logger.info("Skipping: Empty split")
                
                ## Update evaluation data

                logger.info("Update experiment data")
                eval_data["status"]["progress"][dataset_name][split_str] = True
                goripy.file.json.save_json(eval_data, eval_data_filename)
                
                continue
            
            # Prepare subprocess arguments

            dataset_split_idx_chunk_list = goripy.array.chunk.chunk_partition_num(dataset_split_idxs, world_size)

            pred_cat_logits_temp_filename_list = [
                os.path.join(pred_cat_logits_dataset_dirname, "{:s}__{:d}.pt".format(split_str, rank))
                for rank in range(world_size)
            ]

            target_cat_probs_temp_filename_list = [
                os.path.join(target_cat_probs_dataset_dirname, "{:s}__{:d}.pt".format(split_str, rank))
                for rank in range(world_size)
            ]

            cat_weights_temp_filename_list = [
                os.path.join(cat_weights_dataset_dirname, "{:s}__{:d}.pt".format(split_str, rank))
                for rank in range(world_size)
            ]

            # Zip subprocess args

            proc_args_zip = zip(
                dataset_split_idx_chunk_list,
                pred_cat_logits_temp_filename_list,
                target_cat_probs_temp_filename_list,
                cat_weights_temp_filename_list
            )

            # Run inference subprocesses

            logger.info("Running inference subprocesses")

            procs = []

            for rank, proc_args in enumerate(proc_args_zip):

                (
                    dataset_split_idx_chunk,
                    pred_cat_logits_temp_filename,
                    target_cat_probs_temp_filename,
                    cat_weights_temp_filename
                ) = proc_args

                proc = multiprocessing.Process(
                    target=compute_cat_logits_probs_subprocess,
                    args=(
                        rank,
                        command_args,

                        logging_settings,
                        cat_metadata,

                        dataset_name,
                        split_str,

                        dataset_split_idx_chunk,
                        pred_cat_logits_temp_filename,
                        target_cat_probs_temp_filename,
                        cat_weights_temp_filename
                    )
                )

                procs.append(proc)

            for proc in procs: proc.start()
            for proc in procs: proc.join()

            for proc_zidx, proc in enumerate(procs):
                if proc.exitcode != 0:
                    raise ValueError("Subprocess {:s} [{:d}] exited with code {:d}".format(
                        "compute_cat_logits_probs_subprocess",
                        proc_zidx,
                        proc.exitcode
                    ))

            logger.info("Finished inference subprocesses")

            # Join all predicted category logits

            logger.info("Join all predicted category logits")

            pred_cat_logits_temp_tenn_list = []

            for pred_cat_logits_temp_filename in pred_cat_logits_temp_filename_list:

                pred_cat_logits_temp_tenn = torch.load(pred_cat_logits_temp_filename, map_location=device)
                pred_cat_logits_temp_tenn_list.append(pred_cat_logits_temp_tenn)
                os.remove(pred_cat_logits_temp_filename)

                logger.info("Deleted temp predicted category logits {:s}".format(pred_cat_logits_temp_filename))

            pred_cat_logits_joined_tenn = torch.cat([
                pred_cat_logits_temp_tenn
                for pred_cat_logits_temp_tenn in pred_cat_logits_temp_tenn_list
            ])
            
            pred_cat_logits_joined_filename = os.path.join(pred_cat_logits_dataset_dirname, "{:s}.pt".format(split_str))
            torch.save(pred_cat_logits_joined_tenn, pred_cat_logits_joined_filename)

            del pred_cat_logits_temp_tenn_list
            del pred_cat_logits_joined_tenn

            logger.info("Saved joined predicted category logits {:s}".format(pred_cat_logits_joined_filename))

            # Join all target category probabilities

            logger.info("Join all target category probabilities")

            target_cat_probs_temp_tenn_list = []

            for target_cat_probs_temp_filename in target_cat_probs_temp_filename_list:

                target_cat_probs_temp_tenn = torch.load(target_cat_probs_temp_filename, map_location=device)
                target_cat_probs_temp_tenn_list.append(target_cat_probs_temp_tenn)
                os.remove(target_cat_probs_temp_filename)

                logger.info("Deleted temp target category probabilities {:s}".format(target_cat_probs_temp_filename))

            target_cat_probs_joined_tenn = torch.cat([
                target_cat_probs_temp_tenn
                for target_cat_probs_temp_tenn in target_cat_probs_temp_tenn_list
            ])

            target_cat_probs_joined_filename = os.path.join(target_cat_probs_dataset_dirname, "{:s}.pt".format(split_str))
            torch.save(target_cat_probs_joined_tenn, target_cat_probs_joined_filename)

            del target_cat_probs_temp_tenn_list
            del target_cat_probs_joined_tenn

            logger.info("Saved joined target category probabilities {:s}".format(target_cat_probs_joined_filename))

            # Join all category weights

            logger.info("Join all category weights")

            cat_weights_temp_tenn_list = []

            for cat_weights_temp_filename in cat_weights_temp_filename_list:

                cat_weights_temp_tenn = torch.load(cat_weights_temp_filename, map_location=device)
                cat_weights_temp_tenn_list.append(cat_weights_temp_tenn)
                os.remove(cat_weights_temp_filename)

                logger.info("Deleted temp category weights {:s}".format(cat_weights_temp_filename))

            cat_weights_joined_tenn = torch.cat([
                cat_weights_temp_tenn
                for cat_weights_temp_tenn in cat_weights_temp_tenn_list
            ])

            cat_weights_joined_filename = os.path.join(cat_weights_dataset_dirname, "{:s}.pt".format(split_str))
            torch.save(cat_weights_joined_tenn, cat_weights_joined_filename)
            
            del cat_weights_temp_tenn_list
            del cat_weights_joined_tenn

            logger.info("Saved joined category weights {:s}".format(cat_weights_joined_filename))

            # Update evaluation data

            logger.info("Update experiment data")
            eval_data["status"]["progress"][dataset_name][split_str] = True
            goripy.file.json.save_json(eval_data, eval_data_filename)
