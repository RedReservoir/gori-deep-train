# Analysis pipeline: `evaluation_multiattr_outputs`

The `evaluation_multiattr_outputs.py` script is for generating multi-attribute confusion matrix aggregate and metric plots and tables of an `evaluation_multiattr` evaluation run in the `gori-deep-train` project.

## How to Run

Below is an example bash command to run the `evaluation_multiattr_outputs.py` script:

```bash
python \
  evaluation_multiattr_outputs.py \
  <analysis_name> \
  <evaluation_name>
```

The result of running this script is the creation of the following directory:

  - `${GORIDEEPSTYLE_DATA_HOME}/analysis_results/<analysis_name>`: Analysis results directory.

Make sure that the following directories exist prior to run this command:

  - `${GORIDEEPSTYLE_DATA_HOME}/analysis_settings/<analysis_name>`: Analysis settings directory.
  - `${GORIDEEPSTYLE_DATA_HOME}/evaluation_results/<evaluation_name>`: Evaluation results directory.

Resource consumption is low when running this script, and no GPUs are used.

## Analaysis Settings

The analysis settings directory must contain the following settings files:

```text
.
├── groups.json
├── metrics.json
└── outputs.json
```

## Analysis Results

The analysis results directory will have the following contents:

```text
.
├── outputs
│   ├── multiattr_conf_agg_plots
│   └── multiattr_conf_metric_plots
│   └── multiattr_conf_metric_tables
└── settings
```

The `settings` directory is copied over from the analysis settings directory at the beginning of the analysis pipeline. This is done to conserve the original settings in case of accidental modification.

The `outputs` directory contains the results of the analysis pipeline, comprising multiple outputs:

  - `multiattr_conf_agg_plots`: Confusion aggregates plots. Equivalent to mini-confusion matrices for each particular attribute.
  - `multiattr_conf_metric_plots`: Confusion metric plots. Showcases usual confusion matrix metrics such as Precision, Recall, Accuracy, F1-Score, and Fβ-Score, both averaged and per attribute.
  - `multiattr_conf_metric_tables`: Confusion metric tables. Showcases usual confusion matrix metrics such as Precision, Recall, Accuracy, F1-Score, and Fβ-Score, both averaged and per attribute.
