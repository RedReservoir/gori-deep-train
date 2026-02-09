## Checkpoint Saver

The `pymodules/checkpoint_saver.py` Python file defines the checkpoint saver object. When imported, this file must create the following variable:

  - `checkpoint_saver`: An instance of a subclass of `yadscore.checkpoint_savers.base.BaseCheckpointSaver`. This is the checkpoint saver that will be used through the training pipeline.

During the training pipeline, the checkpoint saver defined in this Python module is stored in the `checkpoint_saver` variable.

Data for the checkpoint saver is saved to the `data/checkpoint_saver_data` directory after every train and eval loop. This directory must be created beforehand.
