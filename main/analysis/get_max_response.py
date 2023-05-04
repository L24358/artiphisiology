"""
Should be used after guided_backprop2.

"""

import numpy as np
import handytools.navigator as nav
import handytools.visualizer as vis
import handytools.manipulator as man
import spectools.models.models as mdl
import spectools.visualization as vis2
from torch.utils.data import DataLoader
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
hkey = 3 # layer of interest
top = 10
device = "cuda:0"

# load
mod = mdl.get_alexnet().to(device)
dataset = Imagenette("train")
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
units = nav.npload(nav.datapath, "gbp_AN", f"highFOIunits_hkey={hkey}_thre=0.8.npy")

for unit in units:
    print("Unit ", unit)
    max_to_idx = {} # key: max(R), value: i
    for i, data in enumerate(train_dataloader):
        R = nav.npload(nav.datapath, "results", "gbp_AN", f"R_hkey={hkey}_unit={unit}_idx={i}.npy") # shape=(h,w)
        max_idx = np.asarray(man.argsort2d(R, reverse=True)[:1]) # shape=(top, 2)
        Rtop = man.sorted2d(R, max_idx)[0] # find the singular top value
        max_to_idx[Rtop] = i

    keys = np.array(list(max_to_idx.keys()))
    max_idx2 = np.array(sorted(keys, reverse=True))
    Rmaxs = [max_to_idx[max_idx2[j]] for j in range(top)]
    Rmins = [max_to_idx[max_idx2[-j]] for j in range(1, top+1)]

    count, count2 = 0, 0
    images = np.empty((top,3,227,227))
    ggrads = np.empty((top,3,227,227))
    images2 = np.empty((top,3,227,227))
    ggrads2 = np.empty((top,3,227,227))

    for j, data in enumerate(train_dataloader):
        if j in Rmaxs:
            ggrad = nav.npload(nav.datapath, "results", "gbp_AN", f"ggrad_hkey={hkey}_unit={unit}_idx={j}.npy")
            ggrads[count] = ggrad
            images[count] = data[0]
            Rmaxs.remove(j)
            count += 1
        elif j in Rmins:
            ggrad = nav.npload(nav.datapath, "results", "gbp_AN", f"ggrad_hkey={hkey}_unit={unit}_idx={j}.npy")
            ggrads2[count2] = ggrad
            images2[count2] = data[0]
            Rmins.remove(j)
            count2 += 1
    assert count == top
    assert count2 == top

    vis2.show_img_and_grad_top(top, images, ggrads, f"hkey={hkey}_unit={unit}_top.png", title=f"Layer {mdl.AN_layer[hkey]}, Unit {unit} Top Responsive Images", folders=["gbp_AN"])
    vis2.show_img_and_grad_top(top, images2, ggrads2, f"hkey={hkey}_unit={unit}_bottom.png", title=f"Layer {mdl.AN_layer[hkey]}, Unit {unit} Bottom Responsive Images", folders=["gbp_AN"])