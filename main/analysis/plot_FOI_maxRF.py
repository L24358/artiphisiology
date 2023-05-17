import numpy as np
import matplotlib.pyplot as plt
import handytools.visualizer as vis
import spectools.models.models as mdl
from spectools.stimulus.wyeth import get_stimulus
from spectools.responses import get_response_wrapper, get_drr_wrapper
from spectools.metrics.metrics import responsive
from spectools.models.models import AN_layer, VGG16_layer, ResNet18_layer
from spectools.models.calc import get_RF

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

def get_prs_wrap(mtype, hkeys):
    xn, sz, lw, fg, bg = 227, 50, 1.5, 1.0, 0.0

    # fill stimulus
    fill = get_stimulus(1, xn=xn, sz=sz, lw=lw, fg=fg, bg=bg)
    fname = lambda hkey: f"hkey={hkey}_fill=1_xn={xn}_sz={sz}_lw={lw}_fg={fg}_bg={bg}.npy"
    R_fills = get_response_wrapper(hkeys, fill, fname, mtype=mtype)

    # outline stimulus
    outline = get_stimulus(0, xn=xn, sz=sz, lw=lw, fg=fg, bg=bg)
    fname = lambda hkey: f"hkey={hkey}_fill=0_xn={xn}_sz={sz}_lw={lw}_fg={fg}_bg={bg}.npy"
    R_outlines = get_response_wrapper(hkeys, outline, fname, mtype=mtype)

    prs, resp = get_prs(R_fills, R_outlines, hkeys)
    return prs, resp

# main
device = "cuda:0"
color = vis.color_generator(["r", "g", "b"])
for mtype in ["AN", "VGG16", "VGG16b"]:
    # load model, info and define parameters
    if mtype == "AN": mfunc = mdl.get_alexnet; ldic = mdl.AN_layer
    elif mtype == "VGG16": mfunc = mdl.get_vgg16; ldic = mdl.VGG16_layer
    elif mtype == "VGG16b": mfunc = mdl.get_vgg16b; ldic = mdl.VGG16b_layer
    hkeys = list(ldic.keys())
    model = mfunc(hidden_keys=hkeys).to(device)

    # obtain rf, foi
    prs_an, resp_an = get_prs_wrap(mtype, hkeys)
    rfs = get_RF(model)
    foi = [prs_an[hkey][resp_an[hkey]].mean() for hkey in hkeys]
    maxrf = [rfs[hkey] for hkey in hkeys]

    label = "VGG11" if mtype == "VGG16" else mtype
    label = "VGG16" if mtype == "VGG16b" else mtype
    plt.plot(maxrf, foi, label=label, color=next(color), marker=".")
plt.xlabel("Max. receptive field size (pixels)"); plt.ylabel("FOI"); plt.suptitle("FOI v maximum RF for different networks")
plt.legend()
vis.savefig("FOI_v_maxRF.png")