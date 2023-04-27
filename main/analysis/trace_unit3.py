"""
Store responses to Imagenette for modulated network. Currently used for VGG16 only.

@ TODO:
    - Not finished.
"""

import gc
import numpy as np
import torch
import handytools.navigator as nav
import spectools.basics as bcs
from copy import deepcopy
from torch.utils.data import DataLoader
from spectools.models.models import get_vgg16
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
key = 11
unit = 435 # this is the target unit in layer ``key``
N = 256
bs = 128
device = "cuda:0"
verbose = False
torch.manual_seed(42)
print(f"CAUTION: the pre-layer number of units ``N`` needs to be adjusted manually based on key. Current key is {key}, and N is {N}.")

# define dataset, models
dataset = Imagenette()
train_dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
model = get_vgg16(hidden_keys=[key]).to(device)

start, end = 4, 100 # which batch to start and end with
model.eval()
for i, data in enumerate(train_dataloader): # iterate over batches

    print("Batch ", i)
    if i >= start:

        image, _, _ = data
        model(image.to(device), premature_quit = True)
        R_truth = model.hidden_info[key][0][:, unit, ...].detach().cpu().numpy().squeeze() # shape = (B, C, H, W)
        model.reset_storage()

        for n in range(N): # iterate over units in previous layer

            if verbose: print("Turning off unit ", n)
            model_copy = deepcopy(model)
            model_copy.eval()
            bcs.set_to_zero(model_copy, f"features.{key}.weight", unit, n) # turns off n --> unit connections

            model_copy(image.to(device), premature_quit = True)
            R_leave1 = model_copy.hidden_info[key][0][:, unit, ...].detach().cpu().numpy().squeeze()
            diff = abs(R_leave1 - R_truth) # absolute value
            diff_flat = np.mean(diff.reshape(len(diff), -1), axis=1) # average over image
            nav.npsave(diff_flat, nav.datapath, "results", "subtraction_VGG16", f"imagenette_unit={unit}_key={key}_preunit={n}_B={i}.npy")

            del model_copy
            gc.collect()

    if i > end-1: break



