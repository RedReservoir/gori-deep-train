## Module Transforms

The `pymodules/module_transforms` directory contains multiple Python files with name `<dataset_name>.py`, where `<dataset_name>` is the name of the dataset they are defining module transforms for (see the `pymodules/datasets.md` documentation file). When imported, each of these files must create the following variables:

  - `ModuleTransform`: A subclass of `yadscore.module_transforms.base.BaseModuleTransform`, which processes a data batch using the defined modules in order to generate losses. This class is instantiated during the training pipeline.
  - `loss_ten_reg_key_list`: A list of dicts, which defines which loss tensors from the data batch should be associated to which loss registers after passing through the module transform.

During the training pipeline, the modules defined in these Python modules are stored in the `module_transforms_pool` variable. 

Additionally, the `loss_ten_reg_key_list` variables defined in these modules are stored in the `module_transform_loss_ten_reg_key_list_dict` variable.

---

Schema for the `loss_ten_reg_key_list` list:

```Python
[
  {
    "loss_ten_key": <str>,
    "loss_reg_key": <str>
  },
  ...
]
```

  - `loss_ten_key`: Name of the loss tensor in the data batch.
  - `loss_reg_key`: Name of the loss register to associate the loss tensor to (see the `settings/loss_registers.md` documentation file).
