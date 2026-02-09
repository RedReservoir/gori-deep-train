# Dataloaders

The `pymodules/dataloaders` directory contains multiple Python files with name `<dataset_name>.py`, where `<dataset_name>` is the name of the dataset they are defining dataloaders for (see the `pymodules/datasets.md` documentation file). When imported, each of these files must create the following variables:

  - `dataloader_collate_fn`: A `torch.utils.data.DataLoader` collate function to apply to the dataloader for `<dataset_name>`. Additionally, this function must also set a `"batch_size"` member in the `data_batch` containing the batch size.

  - `dataloader_args`: A dict with arguments for the dataloader.

---

Schema for the `dataloader_args` dict:

```Python
{
    "batch_size": <int>,
    "num_workers": <int>,
    "prefetch_factor": <int>
}
```

  - `batch_size`: Passed to `torch.utils.data.DataLoader`.
  - `num_workers`: Passed to `torch.utils.data.DataLoader`.
  - `prefetch_factor`: Passed to `torch.utils.data.DataLoader`.
