# Groups

The `settings/groups.json` JSON file defines dataset-split groups. A dataset-split group is just a list of dataset and split tuples, the predictions of which to group together for generating plots.

For each dataset-split group, both a user defined group name, and the list of dataset and split tuples must be provided.

Schema:

```json
{
  "<group_name>": [
    {
      "dataset": <str>,
      "split": <str>,
    }
    ...
  ],
  ...
}
```

  - `dataset`: Dataset name.
  - `split`: Split name (`"train"`, `"val"` or `"test"`).
