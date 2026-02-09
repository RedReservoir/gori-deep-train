# Plots

The `settings/plots.json` JSON file defines plot settings. Plot settings must be provided for the total loss plot, the specific loss plots, and the group loss plots.

Schema:

```json
{
  "total_loss": {
    "weighted": <bool>,
    "per_item": <bool>
  },
  "spec_loss": {
    "weighted": <bool>,
    "per_item": <bool>
  },
  "group_loss": [
    {
      "group_name": <str>,
      "weighted": <bool>,
      "per_item": <bool>
    },
    ...
  ]
}
```

  - `group_name`: If applicable, the loss group name to apply the plot settings to.
  - `weighted`: If `true`, the loss at every epoch will be multiplied by its associated loss weight (per epoch). 
  - `per_item`: If `true`, the loss at every epoch will be normalized by dividing by the number of items (data instances). Useful when the number of items changes per epoch.
