## Metrics

The `settings/metrics.json` JSON file defines confusion matrix metrics. 

Schema:

```json
{
  "<metric_name>": {
    "metric_fun_name": <str>,
    "kwargs": {
      ...
    }
  }
  ...
}
```

  - `metric_fun_name`: Metric function to use. To view possible options, check the `metric_fun_factory` method defined in the `evaluation_cat_plots.py` script.
  - `kwargs`: Passed to the metric function.
