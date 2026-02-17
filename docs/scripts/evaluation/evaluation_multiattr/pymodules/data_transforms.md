# Data Transforms

The `pymodules/data_transforms` directory contains multiple Python files with name `<dataset_name>.py`, where `<dataset_name>` is the name of the dataset they are defining data transforms for (see the `pymodules/datasets.md` documentation file). When imported, each of these files must create the following variables and/or methods:

  - `instantiate_data_transform`: A method that returns an instance of a subclass of `gorideep.data_transforms.base.BaseDataTransform`, used to transform data coming from the dataset during the dataset-split evaluation.

---

Signature of `instantiate_data_transform`:

```Python
def instantiate_data_transform(
    logger
):
    ...
```
