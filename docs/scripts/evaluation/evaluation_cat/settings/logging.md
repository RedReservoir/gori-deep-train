## Logging

The `settings/logging.json` JSON file defines logging settings. Schema:

```json
{
  "tqdm": {
    "enabled": <bool>,
    "freq": <int>
  }
}
```

  - `tqdm`: Logging settings for evaluation progress bars.
    - `enabled`: Iff `true`, progress bars will be logged during the evaluation of dataset-split pairs.
    - `freq`: Number of progress updates to perform during evaluation of a dataset-split pair. Ignored if `enabled` is `false`.
