import torch
import numpy as np
import matplotlib.pyplot as plt
import spectools.models.models as mdl
import handytools.visualizer as vis
from copy import deepcopy
from spectools.stimulus.dataloader import Imagenette
from spectools.responses import get_response_wrapper
from spectools.stimulus.wyeth import get_stimulus
from spectools.responses import get_response_wrapper, get_drr_wrapper
from spectools.metrics.metrics import responsive
from spectools.models.models import AN_layer, VGG16b_layer, ResNet18_layer

def get_prs(R_fills, R_outlines, hkeys, verbose=False):
    prs = {}
    respdic = {}
    for hkey in hkeys:
        R_fill = R_fills[hkey]
        R_outline = R_outlines[hkey]

        prs[hkey] = np.array([])
        respdic[hkey] = []
        
        for unit in range(R_fill.shape[0]):
            resp = responsive(R_fill[unit]) and responsive(R_outline[unit])
            if resp:
                pr = np.corrcoef(R_fill[unit], R_outline[unit])[0][1]
                prs[hkey] = np.append(prs[hkey], pr)
                respdic[hkey].append(int(unit))
            else:
                prs[hkey] = np.append(prs[hkey], np.nan)
                if verbose: print(f"Layer {hkey}, unit {unit} not responsive!")
    return prs, respdic

# params
mtype = "VGG16b"
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

# get foi response
xn, sz, lw, fg, bg = 227, 50, 1.5, 1.0, 0.0
fill = get_stimulus(1, xn=xn, sz=sz, lw=lw, fg=fg, bg=bg)
fname = lambda hkey: f"hkey={hkey}_fill=1_xn={xn}_sz={sz}_lw={lw}_fg={fg}_bg={bg}.npy"
R_fills = get_response_wrapper(hkeys, fill, fname, mtype=mtype)
outline = get_stimulus(0, xn=xn, sz=sz, lw=lw, fg=fg, bg=bg)
fname = lambda hkey: f"hkey={hkey}_fill=0_xn={xn}_sz={sz}_lw={lw}_fg={fg}_bg={bg}.npy"
R_outlines = get_response_wrapper(hkeys, outline, fname, mtype=mtype)
prs_an_foi, resp = get_prs(R_fills, R_outlines, hkeys)

fig = plt.figure(figsize=(3*3, 2*3))

for i in range(len(hkeys)):
    hkey = hkeys[i]
    foi = prs_an_foi[hkey][resp[hkey]]
    ti = np.array(colordic[hkey])[resp[hkey]]
    
    ax = fig.add_subplot(2, 3, i+1)
    ax.scatter(ti, foi, color="k", s=10)
    ax.set_title(AN_layer[hkey])

vis.common_label(fig, "Texture index", "FOI")
plt.suptitle("FOI v ColorIndex for AlexNet")
plt.tight_layout()
vis.savefig(f"FOI_CI_mtype={mtype}_seed={seed}.png")