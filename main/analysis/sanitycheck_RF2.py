"""
@ Steps for modifying mdl.ResNet18:
    1. In _forward_impl in class ResNet18, replace pre_layers with [self.conv1, identity, self.relu, self.maxpool], with identity = nn.Identity()
    2. In foward in BasicBlock, comment out all the bn lines.
    3. In the downsample section, replace identity = self.downsample(x) with:
        for child in self.downsample[:1]: identity = child(x)
"""

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import handytools.visualizer as vis
import handytools.manipulator as man
import spectools.basics as bcs
import spectools.models.models as mdl
from math import ceil, floor
from spectools.models.calc import get_RF_resnet
from spectools.responses import get_response_wrapper

# hyperparameters
mtype = "ResNet18"
device = "cpu"

# load model, info and define parameters
mfunc = mdl.get_resnet18; ldic = mdl.ResNet18_layer
hkeys = list(ldic.keys())
model = mfunc(hidden_keys=hkeys).to(device)

# narrowing stimulus
def get_narrowing_stim(pxl, ival):
    center = (pxl-1)/2

    stims = []
    grid = range(1, pxl//2, ival)
    for stride in grid:
        lower = ceil(center - stride)
        upper = floor(center + stride + 1)
        
        stim = np.ones((pxl, pxl))
        stim[lower:upper, lower:upper] = np.zeros((upper-lower, upper-lower))
        stim = np.expand_dims(stim, axis=(0,1))
        stim = np.tile(stim, (1,3,1,1))
        stims.append(stim)
    return np.vstack(stims), grid
    
def combine_dict2(dict1, dict2):
    dict12 = {}
    # Combine dictionaries
    for key in set(dict1.keys()).union(dict2.keys()):
        dict12[key] = dict1.get(key, []) + dict2.get(key, [])
    return dict12

# main
X, grid = get_narrowing_stim(800, 10)
fname = lambda hkey: f"hkey={hkey}_narrowing.npy"
Rcs = get_response_wrapper(hkeys, torch.from_numpy(X), fname, mtype=mtype, save=False, override=True)
rfs = get_RF_resnet()
print(rfs)

count = 0
N = len(hkeys)
n = floor(N/3) + 1
fig = plt.figure(figsize=(3*3, 3*n))
for hkey in hkeys:
    Rc = Rcs[hkey]
    Rmean = np.mean(Rc, axis=0)
    Rstd = np.std(Rc, axis=0)
    target = ceil(rfs[hkey]//2)
    
    ax = fig.add_subplot(n, 3, count+1)
    ax.plot(grid, Rmean, color="k")
    ax.fill_between(grid, Rmean-Rstd, Rmean+Rstd, color="k", alpha=0.2)
    ax.plot([target, target], [min(Rmean-Rstd), max(Rmean+Rstd)], color="b", linestyle="--")
    ax.set_title(ldic[hkey])
    count += 1


vis.common_label(fig, "pixels", "Average response of layer")
plt.suptitle(f"Response as a function of narrowing stimulus for {mtype}")
plt.tight_layout()
vis.savefig(f"sanitycheck_RF_mdl={mtype}.png")
    