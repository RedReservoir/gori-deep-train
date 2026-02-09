## Early Stopper

The `pymodules/early_stopper.py` Python file defines the early stopper object. When imported, this file must create the following variable:

  - `early_stopper`: An instance of a subclass of `yadscore.early_stoppers.base.BaseEarlyStopper`. This is the early stopper that will be used through the training pipeline.

During the training pipeline, the early stopper defined in this Python module is stored in the `early_stopper` variable.

Data for the early stopper is saved to the `data/early_stopper_data` directory after every train and eval loop. This directory must be created beforehand.
