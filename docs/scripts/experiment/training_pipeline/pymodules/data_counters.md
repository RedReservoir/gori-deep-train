## Data Counters

The `pymodules/data_counters` directory contains multiple Python files with name `<data_counter_name>.py`, where `<data_counter_name>` is the name of the data counter they define. When imported, each of these files must create the following variables:

  - `data_counter`: An instance of a subclass of `yadscore.data_counters.base.BaseDataCounter`.
  - `dataset_name_list`: A list of dataset names to iterate through for counting.

During the training pipeline, the data counters defined in these Python modules are stored in the `data_counter_pool` variable.

Data for each data counter is saved to the `data/data_counter_data/<data_counter_name>` directory right at the beginning of the training pipeline, after the data counters have been used to loop through the datasrts. This directory must be created beforehand.

---

Schema for the `dataset_name_list` list:

```Python
[
  <str>,
  ...
]
```

Each element of this list must be a dataset name (see the `pymodules/datasets.md` documentation file).
