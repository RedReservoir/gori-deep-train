# evaluation_multiattr

The `evaluation_multiattr.py` script is for evaluating models on a binary attribute prediction task in the `ya-deep-style` framework. It is a generic pipeline that admits usage of multiple models, datasets and tasks by requiring the user to define these specific components in separate configuration files. Generic multi-attributes can be defined as metadata files.

---

### How to Run

Below is an example bash command to run the `evaluation_multiattr.py` script, which automatically balances prediction workload amongst all GPUs of the host machine:

```bash
python \
  evaluation_multiattr.py \
  <evaluation_name>
```

The result of running this script is the creation of the following directory:

  - `${YADEEPSTYLE_DATA_HOME}/evaluation_results/<evaluation_name>`: Evaluation results directory.

Make sure that the following directories exist prior to run this command:

  - `${YADEEPSTYLE_DATA_HOME}/evaluation_settings/<evaluation_name>`: Evaluation settings directory.
  - `${YADEEPSTYLE_DATA_HOME}/evaluation_logs/<evaluation_name>`: Evaluation logs directory

The `evaluation_multiattr.py` script also accepts two (mutually exclusive) flags:

  - `--resume`: Resumes the (unfinished) evaluation.
  - `--reset`: Resets the evaluation completely, starting over from the beginning. This DELETES the evaluation results and logs directories.

It is recommended to set these environment variables to control which CUDA devices are used:

  - `CUDA_DEVICE_ORDER="PCI_BUS_ID"` for consistent CUDA device ordering.
  - `CUDA_VISIBLE_DEVICES` to select a subset of CUDA devices from the machine.

It is also recommended to set these environment variables so that the machine's CPUs are spread evenly amongst the GPUs:

  - `OMP_NUM_THREADS`
  - `OPENBLAS_NUM_THREADS`
  - `MKL_NUM_THREADS`
  - `VECLIB_MAXIMUM_THREADS`
  - `NUMEXPR_NUM_THREADS`

---

### Evaluation Settings

The evaluation settings directory must have the following contents:

```
.
├── pymodules
│   ├── data_transforms
│   ├── dataloaders
│   ├── datasets
│   ├── module_transforms
│   ├── modules
└── settings
    ├── logging.json
    └── metadata.json
```

The `pymodules` directory contains Python modules defining different components to use during the evaluation pipeline. Specific documentation for each component family can be found in the `pymodules` subdirectory inside this documentation directory.

The `settings` directory contains JSON files with more settings to control the evaluation pipeline. Specific documentation for each settings file can be found in the `settings` subdirectory inside this documentation directory.

Please carefully read the documentation for each of the `pymodules` and `settings`, as many of those files are not independent, and often must contain matching values.

---

### Evaluation Results

The evaluation results directory will have the following contents:

```
.
├── pymodules
├── settings
└── data
    ├── pred_multiattr_logits
    ├── target_multiattr_probs
    └── multiattr_weights
```

The `pymodules` and `settings` directories are copied over from the evaluation settings directory at the beginning of the evaluation pipeline. This is done to prevent issues caused by their accidental modification mid/after evaluation. Thanks to this, running the `evaluation_multiattr.py` script is an idempotent operation.

The `temp_data` directory is used by the `evaluation_multiattr.py` script to store temporary data that is later copied over to the `data` directory. You can safely ignore this directory.

Finally, the `data` directory contains state and record data from many of the components used throughout the evaluation pipeline. This consists of the following subdirectories:

  - `pred_multiattr_logits`: Predicted multi-attribute logits.
  - `target_multiattr_probs`: Target multi-attribute probabilities.
  - `multiattr_weights`: Multi-attribute weights.

Each of these subdirectories will contain a `<dataset_name>/<split_str>.pt` file with the results for each dataset-split pair.
