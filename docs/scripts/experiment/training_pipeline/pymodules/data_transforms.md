# Data Transforms

The `pymodules/data_transforms` directory contains multiple Python files with name `<dataset_name>.py`, where `<dataset_name>` is the name of the dataset they are defining data transforms for (see the `pymodules/datasets.md` documentation file). When imported, each of these files must create the following variables:

  - `instantiate_train_data_transform`: A method that returns an instance of a subclass of `gorideep.data_transforms.base.BaseDataTransform`, used to transform data coming from the dataset in the train loop.
  - `instantiate_eval_data_transform`: A method that returns an instance of a subclass of `gorideep.data_transforms.base.BaseDataTransform`, used to transform data coming from the dataset in the eval loop.

During the training pipeline, the modules defined in these Python modules are stored in the `data_transforms_pool` variable, which contains both `train` and `eval` subdicts.

---

Signature of `instantiate_train_data_transform` and `instantiate_eval_data_transform`:

```Python
def instantiate_train_data_transform(
    logger
):
    ...

def instantiate_data_transform(
    logger
):
    ...
```







