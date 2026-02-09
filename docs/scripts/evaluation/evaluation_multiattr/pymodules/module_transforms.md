## Module Transforms

The `pymodules/module_transforms` directory contains multiple Python files with name `<dataset_name>.py`, where `<dataset_name>` is the name of the dataset they are defining module transforms for (see the `pymodules/datasets.md` documentation file). When imported, each of these files must create the following variables:

  - `ModuleTransform`: A subclass of `gorideep.module_transforms.base.BaseModuleTransform`, which processes a data batch using the defined modules in order to generate losses. This class is instantiated during the evaluation pipeline.
  - `multiattr_key_ddict`: A nested dict, which defines which tensors from the data batch should be associated to which result buffers after passing through the module transform.

---

Schema for the `multiattr_key_ddict` dict:

```Python
{
  "<supattr_name>": {
    "pred_multiattr_logit_ten_key": <str>,
    "target_multiattr_prob_ten_key": <str>,
    "multiattr_weight_ten_key": <str>
  },
  ...
}
```

  - `<supattr_name>`: Multi-attribute super-attribute name. One entry per each super-attribute name is required.
    - `pred_multiattr_logit_ten_key`: Name of the predicted multi-attribute logits tensor in the data batch.
    - `target_multiattr_prob_ten_key`: Name of the target multi-attribute probabilities tensor in the data batch.
    - `multiattr_weight_ten_key`: Name of the multi-attribute weights tensor in the data batch.
