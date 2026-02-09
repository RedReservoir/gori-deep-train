## Plots

The `settings/plots.json` JSON file defines plot settings. Plot settings must be provided for confusion matrix plots, confusion matrix aggregates plots, and confusion matrix metric plots. Multiple configuration sets

Schema:

```json
{
  "multiattr_conf_aggs": [
    {
      "group_names": [
        "<group_name_1>",
        "<group_name_2>",
        ...
      ],
      "pn_normalized": <bool>
    },
    ...
  ],
  "multiattr_conf_metric": [
    {
      "group_names": [
        "<group_name_1>",
        "<group_name_2>",
        ...
      ],
      "metric_names": [
        "<metric_name_1>",
        "<metric_name_2>",
        ...
      ],
      "pn_normalized": <bool>,
      "show_numbers": <bool>
    },
    ...
  ]
}
```

  - `multiattr_conf_aggs`: Confusion matrix aggregates plots. Can contain multiple elements.
    - `group_names`: List of group names for which to plot the confusion matrix aggregates.
    - `pn_normalized`: If `true`, the confusion matrix aggregates will be PN-normalized (normalized wrt. number of positive and negative samples).

  - `multiattr_conf_aggs`: Confusion matrix metric plots (bar plots). Can contain multiple elements.
    - `group_names`: List of group names for which to plot the confusion matrix metrics.
    - `metric_names`: List of metric names to compute and plot, for each group.
    - `pn_normalized`: If `true`, the confusion matrix aggregates will be PN-normalized before computing metrics (normalized wrt. number of positive and negative samples).
    - `show_numbers`: If `true`, the exact metric numbers will be shown on top of each bar.
  