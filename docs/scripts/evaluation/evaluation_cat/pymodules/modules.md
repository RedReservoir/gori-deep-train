## Modules

The `pymodules/modules` directory contains multiple Python files with name `<module_name>.py`, where `<module_name>` is the name of the module they define. When imported, each of these files must create the following variables:

  - `module`: An instance of a subclass of `torch.nn.Module`.
  - `settings`: A dict with module settings.

During the evaluation pipeline, the modules defined in these Python modules are stored in the `module_pool` variable.

---

Schema for the `settings` dict:

```Python
{
    "load_weights_from_ckp": {
        "ckp_name": <str>
    },
    "load_weights_from_exp": {
        "exp_name": <str>,
        "ckp_pool_name": <str>,
        "module_name": <str>
    }
}
```

  - `load_weights_from_ckp`: Load initial model weights from a model checkpoint file. Ignored if not provided.
    - `ckp_name`: Name of the checkpoint file.
  - `load_weights_from_exp`: Load initial model weights from a previous experiment. Ignored if not provided.
    - `exp_name`: Name of the previous experiment.
    - `ckp_pool_name`: Name of the checkpoint pool directory inside the experiment results.
    - `module_name`: Name of the module in the previous experiment.
