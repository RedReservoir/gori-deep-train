# Analysis pipeline: `training_pipeline_loss_plots`

The `training_pipeline_loss_plots.py` script is for generating loss plots of an experiment run in the `gori-deep-train` project. The status of the experiment run is not relevant, plots can be generated mid-training.

## How to Run

Below is an example bash command to run the `training_pipeline_loss_plots.py` script:

```bash
python \
  training_pipeline_loss_plots.py \
  <analysis_name> \
  <experiment_name>
```

The result of running this script is the creation of the following directory:

  - `${GORIDEEPSTYLE_DATA_HOME}/analysis_results/<analysis_name>`: Analysis results directory.

Make sure that the following directories exist prior to run this command:

  - `${GORIDEEPSTYLE_DATA_HOME}/analysis_settings/<analysis_name>`: Analysis settings directory.
  - `${GORIDEEPSTYLE_DATA_HOME}/experiment_results/<experiment_name>`: Experiment results directory.

Resource consumption is low when running this script, and no GPUs are used.

## Analaysis Settings

The analysis settings directory must contain the following settings files:

```text
.
├── groups.json
└── plots.json
```

## Analysis Results

The analysis results directory will have the following contents:

```text
.
├── data
│   ├── group_loss_plots
│   ├── spec_loss_plots
│   └── total_loss_plot.jpg
└── settings
```

The `settings` directory is copied over from the analysis settings directory at the beginning of the analysis pipeline. This is done to conserve the original settings in case of accidental modification.

The `data` directory contains the results of the analysis pipeline, comprising multiple plots visualizing train and validation split loss trends.

  - `group_loss_plots`: Group loss plots.
  - `spec_loss_plots`: Specific loss plots.
  - `total_loss_plot.jpg`: Total loss plot.
