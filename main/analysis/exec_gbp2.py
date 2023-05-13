"""
Should be used after exec_gbp.py.
"""

import numpy as np
import handytools.navigator as nav
import handytools.visualizer as vis
import handytools.manipulator as man
import spectools.models.models as mdl
import spectools.visualization as vis2
from copy import deepcopy
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
hkey = 6
top = 10
plot = True
mtype = "ResNet18"
device = "cuda:0"
units = range(1) #range(mdl.get_units(mtype, hkey))

# load
if mtype == "AN": mfunc = mdl.get_alexnet
elif mtype == "VGG16": mfunc = mdl.get_vgg16
elif mtype == "ResNet18": mfunc = mdl.get_resnet18
mod = mfunc(hidden_keys=[hkey]).to(device)
dataset = Imagenette("train")

# main
max_resp_dic = {} # key:unit, value: [coordinate index, R]
for unit in units:
    print("Unit ", unit)
    max_to_idx = {} # key: max(R), value: i
    idx_to_coor = {} # key: i, value: index of the most resp. coordinate

    for i in range(len(dataset)):
        image, _, _ = dataset[i]
        image = image.unsqueeze(0)
        R = nav.npload(nav.resultpath, f"gbp_{mtype}", f"R_hkey={hkey}_unit={unit}_idx={i}.npy") # shape=(h,w)
        max_idx = np.asarray(man.argsort2d(R, reverse=True)[:1]) # shape=(top=1, 2), coordinate of the top value
        Rtop = man.sorted2d(R, max_idx)[0] # find the singular top value
        max_to_idx[Rtop] = i
        idx_to_coor[i] = max_idx

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

        for j in range(len(dataset)):
            image, _, _ = dataset[j]
            image = image.unsqueeze(0)
            if j in Rmaxs:
                idx1 = Rmaxs_copy.index(j)
                ggrad = nav.npload(nav.resultpath, f"gbp_{mtype}", f"ggrad_hkey={hkey}_unit={unit}_idx={j}.npy")
                ggrads[idx1] = ggrad
                images[idx1] = image[0]
                Rmaxs.remove(j) # removed for efficiency purposes
            elif j in Rmins:
                idx2 = Rmins_copy.index(j)
                ggrad = nav.npload(nav.resultpath, f"gbp_{mtype}", f"ggrad_hkey={hkey}_unit={unit}_idx={j}.npy")
                ggrads2[idx2] = ggrad
                images2[idx2] = image[0]
                Rmins.remove(j)

        # plot top ``top`` images
        vis2.show_img_and_grad_top(top, images, ggrads, f"hkey={hkey}_unit={unit}_top.png", title=f"Layer {mdl.AN_layer[hkey]}, Unit {unit} Top Responsive Images", folders=[f"gbp_{mtype}"])
        # vis2.show_img_and_grad_top(top, images2, ggrads2, f"hkey={hkey}_unit={unit}_bottom.png", title=f"Layer {mdl.AN_layer[hkey]}, Unit {unit} Bottom Responsive Images", folders=[f"gbp_{mtype}"])

    
