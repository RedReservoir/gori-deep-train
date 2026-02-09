# Getting Started

## Python dependencies

The required dependencies for running any scripts from the `gori-deep-train` project are the following:

  - `gori-py-utils` Python package: Generic Python utilities for `gori` projects.
  - `gori-deep-train-core` Python package: Core libraries for the `gori-deep-train` project.

You can simply install them from their respective GitHub repositories:

```bash
pip install git+https://github.com/RedReservoir/gori-py-utils
pip install git+https://github.com/RedReservoir/gori-deep-train
```

<font color="red">**WARNING**</font>: There are many other dependencies necessary for using this project, such as `numpy` and `torch`, which are not listed here.

## The `gori-deep-train` data directory

The `gori-deep-train` project will read and write data from the `gori-deep-train` data directory. Set the `GORIDEEPTRAIN_DATA_HOME` environment variable to the path to this directory, which must exist in your local filesystem.

The `gori-deep-train` data directory must contain the following subdirectories:

```text
.
├── analysis_results
├── analysis_settings
├── evaluation_logs
├── evaluation_results
├── evaluation_settings
├── experiment_logs
├── experiment_results
├── experiment_settings
├── experiment_torchelastic_error_files
├── metadata
└── module_ckps
```

  - `metadata`: Metadata files for metadata classes.
  - `module_ckps`: Miscellaneous module checkpoints.
  - `evaluation_logs`: Log files for evaluation pipelines.
  - `evaluation_results`: Result files for evaluation pipelines.
  - `evaluation_settings`: Setting files for evaluation pipelines.
  - `experiment_logs`: Log files for training pipelines.
  - `experiment_results`: Result files for training pipelines.
  - `experiment_settings`: Setting files for training pipelines.
  - `experiment_torchelastic_error_files`: Torchelastic error files for training pipelines.
  - `analysis_results`: Result files for analysis pipelines.
  - `analysis_settings`: Setting files for analysis pipelines.
