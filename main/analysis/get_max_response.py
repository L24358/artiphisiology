"""
Should be used after guided_backprop2.
"""

import numpy as np
import handytools.navigator as nav
import handytools.visualizer as vis
import handytools.manipulator as man
import spectools.models.models as mdl
import spectools.visualization as vis2
from copy import deepcopy
from torch.utils.data import DataLoader
from spectools.stimulus.dataloader import Imagenette
from spectools.models.calc import get_RF_wrap

# hyperparameters
hkey = 6 # layer of interest
top = 10
plot = True
device = "cuda:0"
units = range(mdl.AN_units[hkey]) #nav.npload(nav.datapath, "gbp_AN", f"highFOIunits_hkey={hkey}_thre=0.8.npy") 

# load
mod = mdl.get_alexnet().to(device)
dataset = Imagenette("train")
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

max_resp_dic = {} # key:unit, value: [coordinate index, R]
for unit in units:
    print("Unit ", unit)
    max_to_idx = {} # key: max(R), value: i
    idx_to_coor = {} # key: i, value: index of the most resp. coordinate
    for i, data in enumerate(train_dataloader):
        R = nav.npload(nav.datapath, "results", "gbp_AN", f"R_hkey={hkey}_unit={unit}_idx={i}.npy") # shape=(h,w)
        max_idx = np.asarray(man.argsort2d(R, reverse=True)[:1]) # shape=(top=1, 2), coordinate of the top value
        Rtop = man.sorted2d(R, max_idx)[0] # find the singular top value
        max_to_idx[Rtop] = i
        idx_to_coor[i] = max_idx

    # sort responses
    keys = np.array(list(max_to_idx.keys())) # all responses
    max_idx2 = np.array(sorted(keys, reverse=True)) # sorted responses

    # save most resp. ranking and location
    rank = [max_to_idx[idx] for idx in max_idx2] # ranking of images
    nav.npsave(rank, nav.datapath, "results", "gbp_AN", f"rank_hkey={hkey}_unit={unit}.npy")
    nav.pklsave(idx_to_coor, nav.datapath, "results", "gbp_AN", f"loc_hkey={hkey}_unit={unit}.pkl")

    if plot:
        Rmaxs = [max_to_idx[max_idx2[j]] for j in range(top)] # top responses
        Rmins = [max_to_idx[max_idx2[-j]] for j in range(1, top+1)] # bottom responses
        Rmaxs_copy = list(deepcopy(Rmaxs))
        Rmins_copy = list(deepcopy(Rmins))

        images = np.empty((top,3,227,227))
        ggrads = np.empty((top,3,227,227))
        images2 = np.empty((top,3,227,227))
        ggrads2 = np.empty((top,3,227,227))

        for j, data in enumerate(train_dataloader):
            if j in Rmaxs:
                idx1 = Rmaxs_copy.index(j)
                ggrad = nav.npload(nav.datapath, "results", "gbp_AN", f"ggrad_hkey={hkey}_unit={unit}_idx={j}.npy")
                ggrads[idx1] = ggrad
                images[idx1] = data[0]
                Rmaxs.remove(j) # removed for efficiency purposes
            elif j in Rmins:
                idx2 = Rmins_copy.index(j)
                ggrad = nav.npload(nav.datapath, "results", "gbp_AN", f"ggrad_hkey={hkey}_unit={unit}_idx={j}.npy")
                ggrads2[idx2] = ggrad
                images2[idx2] = data[0]
                Rmins.remove(j)

        # plot top ``top`` images
        vis2.show_img_and_grad_top(top, images, ggrads, f"hkey={hkey}_unit={unit}_top.png", title=f"Layer {mdl.AN_layer[hkey]}, Unit {unit} Top Responsive Images", folders=["gbp_AN"])
        vis2.show_img_and_grad_top(top, images2, ggrads2, f"hkey={hkey}_unit={unit}_bottom.png", title=f"Layer {mdl.AN_layer[hkey]}, Unit {unit} Bottom Responsive Images", folders=["gbp_AN"])

    
