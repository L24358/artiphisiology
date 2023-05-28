import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import handytools.visualizer as vis
import spectools.basics as bcs
import spectools.models.models as mdl
from math import ceil, floor
from spectools.models.calc import get_RF

# hyperparameters
mtype = "VGG16b"
device = "cuda:0"

# load model, info and define parameters
if mtype == "AN": mfunc = mdl.get_alexnet; ldic = mdl.AN_layer
elif mtype == "VGG16": mfunc = mdl.get_vgg16; ldic = mdl.VGG16_layer
elif mtype == "VGG16b": mfunc = mdl.get_vgg16b; ldic = mdl.VGG16b_layer
hkeys = list(ldic.keys())[:-3] # ignore FC layers
model = mfunc(hidden_keys=hkeys).to(device)

# narrowing stimulus
def get_narrowing_stim(pxl, ival):
    center = (pxl-1)/2

    stims = []
    for stride in range(1, pxl//2, ival):
        lower = ceil(center - stride)
        upper = floor(center + stride + 1)
        
        stim = np.ones((pxl, pxl))
        stim[lower:upper, lower:upper] = np.zeros((upper-lower, upper-lower))
        stim = np.expand_dims(stim, axis=(0,1))
        stim = np.tile(stim, (1,3,1,1))
        stims.append(stim)
    return np.vstack(stims)
    
# main
X = get_narrowing_stim(227, 1)
model(torch.from_numpy(X).to(device))
rfs = get_RF(model)
print(rfs)

count = 0
N = len(hkeys)
n = floor(N/3) + 1
fig = plt.figure(figsize=(3*3, 3*n))
for hkey in hkeys:
    R = model.hidden_info[hkey][0].cpu()
    Rc = bcs.get_center_response(R)
    Rmean = np.mean(Rc, axis=0)
    Rstd = np.std(Rc, axis=0)
    target = ceil(rfs[hkey]//2)
    
    ax = fig.add_subplot(n, 3, count+1)
    ax.plot(Rmean, color="k")
    ax.fill_between(range(len(Rmean)), Rmean-Rstd, Rmean+Rstd, color="k", alpha=0.2)
    ax.plot([target, target], [min(Rmean-Rstd), max(Rmean+Rstd)], color="b", linestyle="--")
    ax.set_title(ldic[hkey])
    count += 1

vis.common_label(fig, "pixels", "Average response of layer")
plt.suptitle(f"Response as a function of narrowing stimulus for {mtype}")
plt.tight_layout()
vis.savefig(f"sanitycheck_RF_mdl={mtype}.png")
    