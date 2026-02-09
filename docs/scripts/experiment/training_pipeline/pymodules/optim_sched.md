## Optimizers and Schedulers

The `pymodules/optim_sched` directory contains multiple Python diles with name `<optim_sched_name>.py`, where `<optim_sched_name>` is the name of the optimizer-scheduler pair they define. When imported, each of these files must create the following methods:

  - `create_optimizer`: A method that takes the `module_pool` as input and returns an optimizer object (an instance of a `torch.optim.Optimizer` subclass). All parameter groups passed to the optimizer must have an additional field called `"name"`.

  - `create_scheduler`: A method that takes the previously created `optimizer` as input and returns a LR Scheduler object (an instance of a `yadscore.schedulers.base.BaseLRScheduler subclass).

  - `scheduler_start_epoch`: The `start_epoch` value to pass to the LR scheduler during initialization.

During the training pipeline, the optimizers and schedulers defined in this Python module are stored in the `optimizer_pool` and `scheduler_pool` variables, respectively. The link relationships between optimizers and schedulers are one-to-one, and module names must not overlap between different optimizer-scheduler pairs.

Checkpoints for each optimizer are saved into to `data/optim_ckps/<optim_sched_name>.pt` files after every train loop. The `data/optim_ckps` directory must be created beforehand.

Data for each scheduler is saved to the `data/sched_data/<optim_sched_name>` subdirectory after every train loop. The `data/sched_data` directory must be created beforehand. Additionally, evolution data of the learning rates for all parameter groups will be saved to `data/lr_data/<optim_sched_name>` subdirectory after every train loop.
