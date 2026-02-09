# Groups

The `settings/groups.json` JSON file defines loss groups. A loss group is just a list of loss register keys to group together as a single loss when generating plots. The total loss of the loss group is the addition of all associated losses.

For each loss group, both a user defined group name, and the list of loss register keys must be provided.

Schema:

```json
{
  "<group_name>": [
    "<loss_reg_key_1>",
    "<loss_reg_key_2>",
    ...
  ],
  ...
}
```
