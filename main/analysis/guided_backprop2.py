"""
Plot the guided backprop images for a particular unit.
"""

import torch
import numpy as np
import handytools.navigator as nav
import handytools.visualizer as vis
import spectools.models.models as mdl
from torch.autograd import Variable
from torch.utils.data import DataLoader
from spectools.models.deconvnet import GuidedBackprop
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
hkey = 3 # layer of interest
bs = 1
device = "cuda:0"

# load
mod = mdl.get_alexnet().to(device)
GBP = GuidedBackprop(mod, device=device)
dataset = Imagenette("train")
train_dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
units = nav.npload(nav.datapath, "gbp_AN", f"highFOIunits_hkey={hkey}_thre=0.8.npy")

def save_gbp(unit):
    print("Unit ", unit)
    for i, data in enumerate(train_dataloader):
        image, _, _ = data
        tt_var = Variable(image.to(device), requires_grad=True)
        ggrads = GBP.generate_gradients(tt_var, hkey, unit)
        R = GBP.outputs[0,unit].cpu().detach().numpy() # shape=(1, #unit, h, w) --> (h,w)
        nav.npsave(R, nav.datapath, "results", "gbp_AN", f"R_hkey={hkey}_unit={unit}_idx={i}.npy")
        nav.npsave(ggrads, nav.datapath, "results", "gbp_AN", f"ggrad_hkey={hkey}_unit={unit}_idx={i}.npy")    
        vis.print_batch(i, 1000)

for unit in units: save_gbp(unit)