# Training Pipeline

The `training_pipeline.py` script is the main script for training models in the `gori-deep-train` project. It is a generic pipeline that admits usage of multiple models, datasets and tasks by requiring the user to define these specific components in separate configuration files.

## How to Run

The `training_pipeline.py` script was written with `torchrun` usage in mind. Below is an example bash command to run this script, using 1 node and 4 processes per node (1 machine with 4 GPUs).

```bash
torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --node_rank=0 \
  --log-dir=${GORIDEEPSTYLE_DATA_HOME}/experiment_logs/<experiment_name> \
  --redirects 3 \
  training_pipeline.py \
    <experiment_name>
```

The result of running this script is the creation of the following directory:

  - `${GORIDEEPSTYLE_DATA_HOME}/experiment_results/<experiment_name>`: Experiment results directory.

Make sure that the following directories exist prior to run this command:

  - `${GORIDEEPSTYLE_DATA_HOME}/experiment_settings/<experiment_name>`: Experiment settings directory.
  - `${GORIDEEPSTYLE_DATA_HOME}/experiment_logs/<experiment_name>`: Experiment logs directory

The `training_pipeline.py` script also accepts two (mutually exclusive) flags:

  - `--resume`: Resumes the (unfinished) experiment from the latest epoch and stage.
  - `--reset`: Resets the experiment completely, starting over from the beginning. This DELETES the experiment results and logs directories.

It is recommended to set these environment variables to control which CUDA devices are used:

  - `CUDA_DEVICE_ORDER="PCI_BUS_ID"` for consistent CUDA device ordering.
  - `CUDA_VISIBLE_DEVICES` to select a subset of CUDA devices from the machine.

It is also recommended to set these environment variables so that the machine's CPUs are spread evenly amongst the GPUs:

  - `OMP_NUM_THREADS`
  - `OPENBLAS_NUM_THREADS`
  - `MKL_NUM_THREADS`
  - `VECLIB_MAXIMUM_THREADS`
  - `NUMEXPR_NUM_THREADS`

You can also set the `TORCHELASTIC_ERROR_FILE` environment variable, which will get uncaught exceptions and trace information to a file. Recommended path: `${GORIDEEPSTYLE_DATA_HOME}/experiment_torchelastic_error_files/<experiment_name>/error.json`. Again, make sure that the following directory exists:

  - `${GORIDEEPSTYLE_DATA_HOME}/experiment_torchelastic_error_files/<experiment_name>`: Experiment torchelastic error files.

## Experiment Settings

The experiment settings directory must have the following contents:

```
.
├── pymodules
│   ├── data_counters
│   ├── data_transforms
│   ├── dataloaders
│   ├── datasets
│   ├── module_transforms
│   ├── modules
│   ├── optim_sched
│   ├── checkpoint_saver.py
│   ├── early_stopper.py
│   └── loss_weighter.py
└── settings
    ├── data_loading.json
    ├── logging.json
    └── loss_registers.json
```

The `pymodules` directory contains Python modules defining different components to use during the training pipeline. Specific documentation for each component family can be found in the `pymodules` subdirectory inside this documentation directory.

The `settings` directory contains JSON files with more settings to control the training pipeline. Specific documentation for each settings file can be found in the `settings` subdirectory inside this documentation directory.

Please carefully read the documentation for each of the `pymodules` and `settings`, as many of those files are not independent, and often must contain matching values.

## Experiment Results

The experiment settings directory will have the following contents:

```
.
├── pymodules
├── settings
├── data
│   ├── data_counter_data
│   ├── module_ckps
│   │   ├── frzn
│   │   ├── init
│   │   ├── last
│   │   ├── best
│   │   └── epoch_<epoch_num>
│   ├── optim_ckps
│   ├── sched_data
│   ├── loss_reg_data
│   ├── loss_weighter_data
│   ├── loss_weight_data
│   ├── early_stopper_data
│   └── checkpoint_saver_data
└── temp_data
```

The `pymodules` and `settings` directories are copied over from the experiment settings directory at the beginning of the training pipeline. This is done to prevent issues caused by their accidental modification mid/after training. Thanks to this, running the `training_pipeline.py` script is an idempotent operation.

The `temp_data` directory is used by the `training_pipeline.py` to store temporary data that is later copied over to the `data` directory. You can safely ignore this directory.

Finally, the `data` directory contains state and record data from many of the components used throughout the training pipeline. Please carefully read the documentation for each of the `pymodules` and `settings` to understand which are exactly the contents of all `data` subdirectories.
