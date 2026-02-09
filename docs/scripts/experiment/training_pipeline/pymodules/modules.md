# Modules

The `pymodules/modules` directory contains multiple Python files with name `<module_name>.py`, where `<module_name>` is the name of the module they define. When imported, each of these files must create the following variables:

  - `module`: An instance of a subclass of `torch.nn.Module`.
  - `settings`: A dict with module settings.

During the training pipeline, the modules defined in these Python modules are stored in the `module_pool` variable.

Checkpoints for each module are saved to files named `<nodule_name>.pt` in the following subdirectories of `data/module_ckps` after:

  - `frzn`: Frozen module checkpoints. Updated at the beginning of the training pipeline.
  - `init`: Non-frozen module initial checkpoints. Updated at the beginning of the training pipeline.
  - `last`: Non-frozen module last checkpoints. Updated after every train and eval loop.
  - `best`: Non-frozen module best checkpoints. Updated after every eval loop.
  - `epoch_<epoch_num>`: Non-frozen module checkpoints at epoch # `<epoch_num>`. Updated after every eval loop.

---

Schema for the `settings` dict:

```Python
{
    "frozen": <bool>,
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

  - `frozen`: Dictates whether the model is subject to optimization. Possible options:
    - `false` (default): The model parameters will be updated during training, and will be saved to checkpoint files.
    - `true`: The model parameters will NOT be updated during training, and will NOT be saved to checkpoint files.
  - `load_weights_from_ckp`: Load initial model weights from a model checkpoint file. Ignored if not provided.
    - `ckp_name`: Name of the checkpoint file.
  - `load_weights_from_exp`: Load initial model weights from a previous experiment. Ignored if not provided.
    - `exp_name`: Name of the previous experiment.
    - `ckp_pool_name`: Name of the checkpoint pool directory inside the experiment results.
    - `module_name`: Name of the module in the previous experiment.
