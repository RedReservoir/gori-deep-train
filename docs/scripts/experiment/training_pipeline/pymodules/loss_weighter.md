# Loss Weighter 

The `pymodules/loss_weighter.py` Python file defines the loss weighter object. When imported, this file must create the following variable:

  - `loss_weighter`: An instance of a subclass of `gorideep.loss_weighters.base.BaseLossWeighter`. This is the loss weighter that will be used through the training pipeline.

During the training pipeline, the loss weighter defined in this Python module is stored in the `loss_weighter` variable.

Data for the checkpoint saver is saved to the `data/loss_weighter_data` directory after every train and eval loop. This directory must be created beforehand.

Additionally, the final loss weight values for every epoch are saved to `data/loss_weight_data/loss_weights_<epoch_num>` files after every eval loop, where `<epoch_num>` is the epoch number. The `data/loss_weight_data` directory must be created beforehand.
