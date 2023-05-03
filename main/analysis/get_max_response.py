"""
Should be used after guided_backprop2.

"""

import torch
import numpy as np
import handytools.navigator as nav
import handytools.visualizer as vis
import handytools.manipulator as man
import spectools.models.models as mdl
import spectools.visualization as vis2
from torch.utils.data import DataLoader
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
hkey = 8 # layer of interest
unit = 197
top = 10
device = "cuda:0"

# load
mod = mdl.get_alexnet().to(device)
dataset = Imagenette("train")
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

max_to_idx = {} # key: max(R), value: i
for i, data in enumerate(train_dataloader):
    R = nav.npload(nav.datapath, "results", "gbp_AN", f"R_hkey={hkey}_unit={unit}_idx={i}.npy") # shape=(h,w)
    max_idx = np.asarray(man.argsort2d(R, reverse=True)[:1]) # shape=(top, 2)
    Rtop = man.sorted2d(R, max_idx)[0] # find the singular top value
    max_to_idx[Rtop] = i

keys = np.array(list(max_to_idx.keys()))
max_idx2 = np.array(sorted(keys, reverse=True))
Rmaxs = [max_to_idx[max_idx2[j]] for j in range(top)]
for j, data in enumerate(train_dataloader):
    if j in Rmaxs:
        image = data[0]
        ggrad = nav.npload(nav.datapath, "results", "gbp_AN", f"ggrad_hkey={hkey}_unit={unit}_idx={j}.npy")
        vis2.show_img_and_grad(image.squeeze(), ggrad, f"temp{j}.png", f"Layer {mdl.AN_layer[hkey]}, Unit {unit}")
        Rmaxs.remove(j)
