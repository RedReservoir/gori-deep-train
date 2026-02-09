# evaluation_multiattr_plots

The `evaluation_multiattr_plots.py` script is for generating multi-attribute confusion matrix aggregate and metric plots of an `evaluation_multiattr` evaluation run in the `ya-deep-style` framework.

---

### How to Run

Below is an example bash command to run the `evaluation_multiattr_plots.py` script:

```bash
python \
  evaluation_multiattr_plots.py \
  <analysis_name> \
  <evaluation_name>
```

The result of running this script is the creation of the following directory:

  - `${YADEEPSTYLE_DATA_HOME}/analysis_results/<analysis_name>`: Analysis results directory.

Make sure that the following directories exist prior to run this command:

  - `${YADEEPSTYLE_DATA_HOME}/analysis_settings/<analysis_name>`: Analysis settings directory.
  - `${YADEEPSTYLE_DATA_HOME}/evaluation_results/<evaluation_name>`: Evaluation results directory.

Resource consumption is low when running this script, and no GPUs are used.

---

### Analaysis Settings

The analysis settings directory must contain the following settings files:

```
.
├── groups.json
├── metrics.json
└── plots.json
```

---

### Analysis Results

The analysis results directory will have the following contents:

```
.
├── data
│   ├── multiattr_conf_aggs_plots
│   └── multiattr_conf_metric_plots
└── settings
```

The `settings` directory is copied over from the analysis settings directory at the beginning of the analysis pipeline. This is done to conserve the original settings in case of accidental modification.

The `data` directory contains the results of the analysis pipeline, comprising multiple plots:

  - `multiattr_conf_aggs_plots`: Confusion matrix aggregates plots. Equivalent to mini-confusion matrices for each particular attribute.
  - `multiattr_conf_metric_plots`: Confusion matrix metric plots. Showcases usual confusion matrix metrics such as Precision, Recall, Accuracy, F1-Score, and Fβ-Score, both averaged and per attribute.
