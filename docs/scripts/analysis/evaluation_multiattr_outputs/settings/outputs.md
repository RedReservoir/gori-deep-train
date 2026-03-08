# Outputs

The `settings/outputs.json` JSON file defines output settings. Output settings must be provided for confusion aggregate plots, confusion metric plots, and confusion metric tables. Multiple configuration sets can be provided for each type of output.

Schema:

```json
{
  "multiattr_conf_agg_plots": [
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
  "multiattr_conf_metric_plots": [
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
  ],
  "multiattr_conf_metric_tables": [
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
      "average": <str>,
      "pn_normalized": <bool>
    },
    ...
  ]
}
```

  - `multiattr_conf_agg_plots`: Confusion aggregate plots. Can contain multiple elements.
    - `group_names`: List of group names for which to generate the confusion aggregate plots.
    - `pn_normalized`: If `true`, the confusion aggregates will be PN-normalized (normalized wrt. number of positive and negative samples).

  - `multiattr_conf_metric_plots`: Confusion metric (bar) plots. Can contain multiple elements.
    - `group_names`: List of group names for which to generate the confusion metric plots.
    - `metric_names`: List of metric names to compute and plot, for each group.
    - `pn_normalized`: If `true`, the confusion aggregates will be PN-normalized before computing metrics (normalized wrt. number of positive and negative samples).
    - `show_numbers`: If `true`, the exact metric numbers will be shown on top of each bar.

  - `multiattr_conf_metric_tables`: Confusion metric tables. Can contain multiple elements.
    - `group_names`: List of group names for which to generate the confusion metric tables.
    - `metric_names`: List of metric names to compute and show in the table, for each group.
    - `average`: Type of metric averaging to use. Options: `"micro"`, `"macro"`.
    - `pn_normalized`: If `true`, the confusion aggregates will be PN-normalized before computing metrics (normalized wrt. number of positive and negative samples).
  