# Utils

The `pymodules/utils` directory (optional) contains any miscellaneous Python files provided by the user. Only Python files in the root level of this directory can be imported.

To import any modules from `pymodules/utils` from any other modules, use the following code snippet:

```Python
import importlib

# This code snippet will load the `pymodules/utils/<pymodule>.py` module.

pymodule = importlib.import_module(
    "utils.<pymodule>"
)
```