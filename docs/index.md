# `gori-deep-train`

Welcome to the `gori-deep-train` project documentation. This project comprises a series of generic training, evaluation and analysis pipelines for deep learning models using PyTorch. The core Python package this project is built upon is the [`gori-deep-train-core`](https://github.com/RedReservoir/gori-deep-train-core) package.

Everything has been written with generalizability in mind, so that it is possible to accomodate for any models, losses, datasets, transforms and more, all under a standard framework. It is possible to extend this project by writing new pipelines using the components defined in the [`gori-deep-train-core`](https://github.com/RedReservoir/gori-deep-train-core) package.

To begin, I recommend checking the [Getting Started](getting-started.md) section, which explains how to setup this project if you wish to use it. After that, you can review the documentation of each script. Scripts are divided into three categories:

  - Experiment: Model training scripts. Only one training script is available.
  - Evaluation: Model evaluation scripts for computing and storing model outputs on evaluation datasets.
  - Analysis: Model analysis scripts for computing metric and loss summaries and plots.
