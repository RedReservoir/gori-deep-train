# Data Transforms

The `pymodules/data_transforms` directory contains multiple Python files with name `<dataset_name>.py`, where `<dataset_name>` is the name of the dataset they are defining data transforms for (see the `pymodules/datasets.md` documentation file). When imported, each of these files must create the following variables:

  - `TrainDataTransform`: A subclass of `gorideep.data_transforms.base.BaseDataTransform`, used to transform data coming from the dataset in the train loop. This class is instantiated during the training pipeline.
  - `EvalDataTransform`: A subclass of `gorideep.data_transforms.base.BaseDataTransform`, used to transform data coming from the dataset in the eval loop. This class is instantiated during the training pipeline.

During the training pipeline, the modules defined in these Python modules are stored in the `data_transforms_pool` variable, which contains both `train` and `eval` subdicts.
