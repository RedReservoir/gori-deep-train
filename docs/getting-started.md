# Getting Started

## Python dependencies

The required dependencies for running any scripts from the `gori-deep-train` project are the following Python packages, along with their own dependencies:

  - [`gori-py-utils`](https://github.com/RedReservoir/gori-py-utils)
  - [`gori-deep-train-core`](https://github.com/RedReservoir/gori-deep-train-core)

## The `gori-deep-train` data directory

The `gori-deep-train` project will read and write data from the `gori-deep-train` data directory. You must create this directory in your local filesystem, and also set the `GORIDEEPTRAIN_DATA_HOME` environment variable to the path to this directory.

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
