# Data Transforms

The `pymodules/data_transforms` directory contains multiple Python files with name `<dataset_name>.py`, where `<dataset_name>` is the name of the dataset they are defining data transforms for (see the `pymodules/datasets.md` documentation file). When imported, each of these files must create the following variables:

  - `DataTransform`: A subclass of `gorideep.data_transforms.base.BaseDataTransform`, used to transform data coming from the dataset during the dataset-split evaluation. This class is instantiated during the evaluation pipeline.
