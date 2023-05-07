"""
Which presynaptic units contribute most to the postsynaptic response? (Conditioned on Most-Exciting-Images)

@ Prerequisites:
    - guided_backprop2.py for backprop on every image
    - get_max_response.py for ranking the images by response
"""

import torch
import numpy as np
import handytools.navigator as nav
import spectools.models.models as mdl
from scipy.signal import convolve2d
from torch.utils.data import DataLoader
from spectools.models.calc import get_images_from_loader
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
hkey = 3
top = 20
units = range(192)

# load
model = mdl.get_alexnet(hidden_keys=[hkey-1])
dataset = Imagenette("train")
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
params = mdl.get_parameters("alexnet")
weights = params[f"features.{hkey}.weight"].detach().numpy()
biases = params[f"features.{hkey}.bias"].detach().numpy()

for unit in units:
    print("Unit ", unit)
    rank = nav.npload(nav.datapath, "results", "gbp_AN", f"rank_hkey={hkey}_unit={unit}.npy")
    imgs = get_images_from_loader(train_dataloader, rank[:top], size=(3,227,227))
    model(torch.from_numpy(imgs))
    R = model.hidden_info[hkey-1][0] # shape = (top, #pre, h, w)

    fs = weights[unit].squeeze() # shape = (#pre, filt_h, filt_w)

    idx_to_coor = nav.pklload(nav.datapath, "results", "gbp_AN", f"loc_hkey={hkey}_unit={unit}.pkl")
    coors = np.asarray([idx_to_coor[rank[r]][0] for r in range(top)])

    preidx_to_impact = {} # key: presynaptic index, value: impact on postsynpatic response
    for f in range(len(fs)):
        preidx_to_impact[f] = []
        for r in range(top):
            impact = convolve2d(R[r,f,...].detach().numpy(), fs[f], mode="full") 
            impact = impact[2:-2, 2:-2] # shape = (top, 27, 27) for hkey=3
            rel_impact = impact[coors[r][0]][coors[r][1]] # the impact at the specific location
            preidx_to_impact[f].append(rel_impact)

    nav.pklsave(preidx_to_impact, nav.datapath, "results", "trace_AN", f"impact_hkey={hkey}_unit={unit}.pkl")