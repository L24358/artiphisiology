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
hkey = 6
mtype = "ResNet18"
device = "cuda:0"

# load model
if mtype == "AN": mfunc = mdl.get_alexnet
elif mtype == "VGG16": mfunc = mdl.get_vgg16
elif mtype == "ResNet18": mfunc = mdl.get_resnet18
mod = mfunc(hidden_keys=[hkey]).to(device)
GBP = GuidedBackprop(mod, device=device, is_resnet=True)
dataset = Imagenette("train")
units = range(1) #range(mdl.get_units(mtype, hkey))

def save_gbp(unit):
    print("Unit ", unit)
    for i in range(len(dataset)):
        image, _, _ = dataset[i]
        image = image.unsqueeze(0)
        tt_var = Variable(image.to(device), requires_grad=True)
        ggrads = GBP.generate_gradients(tt_var, hkey, unit)
        R = GBP.outputs[0,unit] # shape=(1, #unit, h, w) --> (h,w)
        
        nav.npsave(R, nav.resultpath, f"gbp_{mtype}", f"R_hkey={hkey}_unit={unit}_idx={i}.npy")
        nav.npsave(ggrads, nav.resultpath, f"gbp_{mtype}", f"ggrad_hkey={hkey}_unit={unit}_idx={i}.npy")    
        vis.print_batch(i, 1000)

for unit in units:
    if True: #not nav.exists(nav.resultpath, f"gbp_{mtype}", f"R_hkey={hkey}_unit={unit}_idx=0.npy"):
        save_gbp(unit)