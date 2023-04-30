"""
Turn off most but a couple selected pre-units.
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
top = 10
device = "cuda:0"
idx_tpe = "filt"
verbose = False
torch.manual_seed(42)

# define dataset, models
dataset = Imagenette("val")
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
model = get_vgg16(hidden_keys=[key]).to(device)
idx = nav.npload(nav.datapath, "results", "subtraction_VGG16", f"{idx_tpe}idx.npy")

start, end = 0, 100 # which batch to start and end with
model.eval()
for i, data in enumerate(dataloader): # iterate over batches

    print("Batch ", i)
    if i >= start:

        image, _, _ = data
        model(image.to(device), premature_quit = True)
        R_truth = model.hidden_info[key][0][:, unit, ...].detach().cpu().numpy().squeeze() # shape = (B, C, H, W)
        model.reset_storage()

        model_copy = deepcopy(model)
        model_copy.eval()

        for n in idx[top+1:]: # all but the top ``top`` are turned off

            if verbose: print("Turning off unit ", n)
            bcs.set_to_zero(model_copy, f"features.{key}.weight", unit, n) # turns off n --> unit connections

        model_copy(image.to(device), premature_quit = True)
        R_leaveN = model_copy.hidden_info[key][0][:, unit, ...].detach().cpu().numpy().squeeze()
        diff = abs(R_leaveN - R_truth) # absolute value
        diff_flat = np.mean(diff.reshape(len(diff), -1), axis=1) # average over image
        nav.npsave(diff_flat, nav.datapath, "results", "subtraction_VGG16", f"imagenette_unit={unit}_key={key}_preunit={idx_tpe}{top}_B={i}.npy")

        del model_copy
        gc.collect()

    if i > end-1: break