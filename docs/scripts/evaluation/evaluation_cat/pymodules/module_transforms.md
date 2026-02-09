## Module Transforms

The `pymodules/module_transforms` directory contains multiple Python files with name `<dataset_name>.py`, where `<dataset_name>` is the name of the dataset they are defining module transforms for (see the `pymodules/datasets.md` documentation file). When imported, each of these files must create the following variables:

  - `ModuleTransform`: A subclass of `gorideep.module_transforms.base.BaseModuleTransform`, which processes a data batch using the defined modules in order to generate losses. This class is instantiated during the evaluation pipeline.
  - `cat_key_dict`: A dict, which defines which tensors from the data batch should be associated to which result buffers after passing through the module transform.

---

Schema for the `cat_key_dict` dict:

```Python
{
  "pred_cat_logit_ten_key": <str>,
  "target_cat_prob_ten_key": <str>,
  "cat_weight_ten_key": <str>
}
```

  - `pred_cat_logit_ten_key`: Name of the predicted category logits tensor in the data batch.
  - `target_cat_prob_ten_key`: Name of the target category probabilities tensor in the data batch.
  - `cat_weight_ten_key`: Name of the category weights tensor in the data batch.
