## Datasets

The `pymodules/datasets` directory contains multiple Python files with name `<dataset_name>.py`, where `<dataset_name>` is the name of the dataset they define. When imported, each of these files must contain:

  - `dataset`: An instance of a subclass of `gorideep.datasets.base.BaseDataset`.

During the evaluation pipeline, the datasets defined in these Python modules are stored in the `dataset_pool` variable.
