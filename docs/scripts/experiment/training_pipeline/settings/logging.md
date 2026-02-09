# Logging

The `settings/logging.json` JSON file defines logging settings. Schema:

```json
{
  "tqdm": {
    "enabled": <bool>,
    "train_freq": <int>,
    "eval_freq": <int>
  }
}
```

  - `tqdm`: Logging settings for train and eval loop progress bars.
    - `enabled`: Iff `true`, progress bars will be logged during the train and eval loops.
    - `train_freq`: Number of progress updates to perform during a training loop. Ignored if `enabled` is `false`.
    - `eval_freq`: Number of progress updates to perform during a eval loop. Ignored if `enabled` is `false`.
