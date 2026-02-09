# Loss Registers

The `settings/loss_registers.json` JSON file defines the loss registers used during the training pipeline. Schema:

```json
{
  "loss_reg_key_list": [
    <str>,
    ...
  ]
}
```

  - `loss_reg_key_list`: A list of loss register names. One `gorideep.utils.loss_register.LossRegister` object will be created for each loss register name.

During the training pipeline, the loss registers listed in this configuration file will be created and stored in the `loss_reg_pool` variable, which contains both `train` and `eval` subdicts.

Data for each loss register is saved to the `data/loss_reg_data/<split>/<loss_reg_key>` subdirectory, where `<split>` may be `train` or `val`. These `data/loss_reg_data/<split>` directories must be created beforehand. Loss register data consists of:

  - Epoch loss values, saved to the `epoch_data.npz` file after every train or eval loop depending on the `<split>`.
  - Step loss values, saved to `step_data_<epoch_num>.npz` files after every train or eval loop depending on the `<split>`, where `<epoch_num>` is the epoch number.
