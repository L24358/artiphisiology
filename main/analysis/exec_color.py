import torch
import numpy as np
import spectools.models.models as mdl
from copy import deepcopy
from spectools.stimulus.dataloader import Imagenette
from spectools.responses import get_response_wrapper

# params
mtype = "AN"
seed = 42
np.random.seed(seed)

# define input
B = 200
dataset = Imagenette("train")
random_idxs = np.random.choice(range(len(dataset)), B)
X = [dataset[idx][0].unsqueeze(0) for idx in random_idxs]
X = torch.vstack(X)

# define models
if mtype == "AN": mfunc = mdl.get_alexnet; ldic = mdl.AN_layer
elif mtype == "VGG16": mfunc = mdl.get_vgg16; ldic = mdl.VGG16_layer
elif mtype == "VGG16b": mfunc = mdl.get_vgg16b; ldic = mdl.VGG16b_layer
hkeys = list(ldic.keys())
model = mfunc(hidden_keys=hkeys)

# permutation
permts = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]

Rcs_all = {}
for hkey in hkeys: Rcs_all[hkey] = [] 

for permt in permts:
    X_copy = deepcopy(X)
    X_permt = X_copy[:, permt]

    permt_str = ":".join([str(i) for i in permt])
    fname = lambda hkey: f"hkey={hkey}_rotatecolor_permt={permt_str}_seed={seed}.npy"
    Rcs = get_response_wrapper(hkeys, X_permt, fname, mtype=mtype, save=True) # key=hkey, value.shape=(#units, B)

    for hkey in hkeys:
        Rcs_all[hkey].append([Rcs[hkey]])

colordic = {}
for hkey in hkeys:
    Rcs = Rcs_all[hkey] = np.vstack(Rcs_all[hkey]) # value.shape = (6, #units, B)
    color_indices = Rcs.std(axis=0).mean(axis=1) # shape = (#units,)
    colordic[hkey] = color_indices



