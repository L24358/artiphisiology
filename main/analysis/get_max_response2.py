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
from copy import deepcopy

# hyperparameters
# AN:6, VGG16b: 19, ResNet18: 8
mtype = "ResNet18"
hkey = 8 # layer of interest
top = 10
plot = True
device = "cuda:0"
units = [35] #nav.npload(nav.resultpath, f"responses_{mtype}", f"hkey={hkey}_unitsCiMax.npy")
path = "/dataloc/images_npy/"

# filter units
for unit in units:
    if nav.exists(nav.graphpath, f"gbp_{mtype}", f"hkey={hkey}_unit={unit}_top.png"):
        units = np.setdiff1d(units, [unit])
print(units)

max_resp_dic = {} # key:unit, value: [coordinate index, R]
for unit in units:
    print("Unit ", unit)
    max_to_idx = {} # key: max(R), value: i
    idx_to_coor = {} # key: i, value: index of the most resp. coordinate
    for i in range(50000):
        try:
            R = nav.npload(nav.resultpath, f"gbp_{mtype}", f"R_hkey={hkey}_unit={unit}_idx={i}.npy") # shape=(h,w)
            max_idx = np.asarray(man.argsort2d(R, reverse=True)[:1]) # shape=(top=1, 2), coordinate of the top value
            Rtop = man.sorted2d(R, max_idx)[0] # find the singular top value
            max_to_idx[Rtop] = i
            idx_to_coor[i] = max_idx
        except:
            print(f"idx {i} not present.") 
        vis.print_batch(i, 1000)

    # sort responses
    keys = np.array(list(max_to_idx.keys())) # all responses
    max_idx2 = np.array(sorted(keys, reverse=True)) # sorted responses

    # save most resp. ranking and location
    rank = [max_to_idx[idx] for idx in max_idx2] # ranking of images
    nav.npsave(rank, nav.resultpath, f"gbp_{mtype}", f"rank_hkey={hkey}_unit={unit}.npy")
    nav.pklsave(idx_to_coor, nav.resultpath, f"gbp_{mtype}", f"loc_hkey={hkey}_unit={unit}.pkl")

    if plot:
        Rmaxs = [max_to_idx[max_idx2[j]] for j in range(top)] # top responses
        Rmins = [max_to_idx[max_idx2[-j]] for j in range(1, top+1)] # bottom responses
        Rmaxs_copy = list(deepcopy(Rmaxs))
        Rmins_copy = list(deepcopy(Rmins))

        images = np.empty((top,3,227,227))
        ggrads = np.empty((top,3,227,227))
        images2 = np.empty((top,3,227,227))
        ggrads2 = np.empty((top,3,227,227))

        for j in range(50000):
            image = np.expand_dims(nav.npload(path, str(j)+".npy"), 0)
            data = torch.from_numpy(image).float()

            if j in Rmaxs:
                idx1 = Rmaxs_copy.index(j)
                ggrad = nav.npload(nav.resultpath, f"gbp_{mtype}", f"ggrad_hkey={hkey}_unit={unit}_idx={j}.npy")
                ggrads[idx1] = ggrad
                images[idx1] = data
                Rmaxs.remove(j) # removed for efficiency purposes
            elif j in Rmins:
                idx2 = Rmins_copy.index(j)
                ggrad = nav.npload(nav.resultpath, f"gbp_{mtype}", f"ggrad_hkey={hkey}_unit={unit}_idx={j}.npy")
                ggrads2[idx2] = ggrad
                images2[idx2] = data
                Rmins.remove(j)
            vis.print_batch(j, 1000)

        # plot top ``top`` images
        vis2.show_img_and_grad_top(top, images, ggrads, f"hkey={hkey}_unit={unit}_top.png", title=f"Layer {mdl.get_names(mtype, hkey)}, Unit {unit} Top Responsive Images", folders=[f"gbp_{mtype}"])
        vis2.show_img_and_grad_top(top, images2, ggrads2, f"hkey={hkey}_unit={unit}_bottom.png", title=f"Layer {mdl.get_names(mtype, hkey)}, Unit {unit} Bottom Responsive Images", folders=[f"gbp_{mtype}"])

    
