# Dataloaders

The `pymodules/dataloaders` directory contains multiple Python files with name `<dataset_name>.py`, where `<dataset_name>` is the name of the dataset they are defining dataloaders for (see the `pymodules/datasets.md` documentation file). When imported, each of these files must create the following variables:

  - `dataloader_collate_fn`: A `torch.utils.data.DataLoader` collate function to apply to the dataloader for `<dataset_name>`.

  - `train_dataloader_args`: A dict with arguments for the train loop dataloader.
  - `eval_dataloader_args`: A dict with arguments for the eval loop dataloader.

During the train and eval loops, the modules defined in these Python modules are stored in the `dataloader_pool` variable.

---

Schema for the `train_dataloader_args` and `eval_dataloader_args` dicts:

```Python
{
    "batch_size": <int>,
    "num_workers": <int>,
    "prefetch_factor": <int>,
    "point_size": <float>
}
```

  - `batch_size`: Passed to `torch.utils.data.DataLoader`.
  - `num_workers`: Passed to `torch.utils.data.DataLoader`.
  - `prefetch_factor`: Passed to `torch.utils.data.DataLoader`.
  - `point_size`: Number of items equivalent to one data point for this dataset.
