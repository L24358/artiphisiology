"""
Plot the guided backprop images for a particular unit for AN.
"""

import torch
import numpy as np
import handytools.navigator as nav
import handytools.visualizer as vis
import spectools.models.models as mdl
from torch.autograd import Variable
from spectools.models.deconvnet import GuidedBackprop

def get_50000_images(idxs=range(50000)):
    path = "/dataloc/images_npy/"
    arrays = [np.expand_dims(nav.npload(path, str(i)+".npy"), 0) for i in idxs]
    return torch.from_numpy(np.vstack(arrays))

# hyperparameters
# AN:8, VGG16b: 19, ResNet18: 8
mtype = "AN"
hkey = 8 # layer of interest
device = "cuda:0"

# load
mod = mdl.get_alexnet().to(device)
GBP = GuidedBackprop(mod, device=device)
path = path = "/dataloc/images_npy/"
units = nav.npload(nav.resultpath, f"responses_{mtype}", f"hkey={hkey}_unitsTiMin.npy")


def save_gbp(unit):
    print("Unit ", unit)
    for i in range(50000):
        image = np.expand_dims(nav.npload(path, str(i)+".npy"), 0)
        image = torch.from_numpy(image).float()

        tt_var = Variable(image.to(device), requires_grad=True)
        ggrads = GBP.generate_gradients(tt_var, hkey, unit)
        R = GBP.outputs[0,unit] #.cpu().detach().numpy() # shape=(1, #unit, h, w) --> (h,w)
        nav.npsave(R, nav.resultpath, f"gbp_{mtype}", f"R_hkey={hkey}_unit={unit}_idx={i}.npy")
        nav.npsave(ggrads, nav.resultpath, f"gbp_{mtype}", f"ggrad_hkey={hkey}_unit={unit}_idx={i}.npy")    
        vis.print_batch(i, 1000)

for unit in units:
    if not nav.exists(nav.resultpath, f"gbp_{mtype}", f"R_hkey={hkey}_unit={unit}_idx=49999.npy"): save_gbp(unit) 