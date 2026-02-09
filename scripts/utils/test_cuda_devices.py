import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch

for env_var_name in [
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES"
]:

    print("{:s}: {:s}".format(env_var_name, os.environ.get(env_var_name, "<None>")))

if torch.cuda.is_available():

    print("CUDA available")
    print("Number of devices: {:d}".format(torch.cuda.device_count()))
    for device_idx in range(torch.cuda.device_count()):
        print("GPU {:d} - {:s}".format(device_idx, torch.cuda.get_device_name(device_idx)))

else:

    print("CUDA not available")
