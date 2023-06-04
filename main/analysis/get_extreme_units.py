import torch
import numpy as np
import handytools.navigator as nav
import handytools.manipulator as man
import spectools.models.models as mdl
from copy import deepcopy
from spectools.stimulus.dataloader import get_50000_images
from spectools.stimulus.wyeth import get_stimulus
from spectools.responses import get_response_wrapper
from spectools.metrics.metrics import responsive

# params (functions depend on hkeys and mfunc being global)
mtype = "AN"
if mtype == "AN": mfunc = mdl.get_alexnet; ldic = mdl.AN_layer
elif mtype == "VGG16": mfunc = mdl.get_vgg16; ldic = mdl.VGG16_layer
elif mtype == "VGG16b": mfunc = mdl.get_vgg16b; ldic = mdl.VGG16b_layer
hkeys = list(ldic.keys())[:-3]

# functions
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

def get_foi():
    xn, sz, lw, fg, bg = 227, 50, 1.5, 1.0, 0.0
    fill = get_stimulus(1, xn=xn, sz=sz, lw=lw, fg=fg, bg=bg)
    fname = lambda hkey: f"hkey={hkey}_fill=1_xn={xn}_sz={sz}_lw={lw}_fg={fg}_bg={bg}.npy"
    R_fills = get_response_wrapper(hkeys, fill, fname, mtype=mtype)
    outline = get_stimulus(0, xn=xn, sz=sz, lw=lw, fg=fg, bg=bg)
    fname = lambda hkey: f"hkey={hkey}_fill=0_xn={xn}_sz={sz}_lw={lw}_fg={fg}_bg={bg}.npy"
    R_outlines = get_response_wrapper(hkeys, outline, fname, mtype=mtype)
    prs_an_foi, resp = get_prs(R_fills, R_outlines, hkeys)
    return prs_an_foi

def get_TI():
    TI, respdic = {}, {}
    for hkey in hkeys:
        Rc = nav.npload(nav.resultpath, f"responses_{mtype}", f"hkey={hkey}_TK_norm=standard.npy") # originally TKRc
        TI[hkey] = []
        respdic[hkey] = []

        for Runit in Rc:
            Rshape = Runit[:225]
            Rtexture = Runit[225:]
            TI[hkey].append(np.std(Rtexture)/np.std(Rshape))

            resp = responsive(Rshape, abs=True) or responsive(Rtexture, abs=True) # it must respond to either texture or shape in abs value
            respdic[hkey].append(resp)
    return TI, respdic

def get_CI():
    seed = 42
    np.random.seed(seed)

    B = 200
    random_idxs = np.random.choice(range(50000), B)
    dataset = get_50000_images(random_idxs)
    X = [dataset[idx].unsqueeze(0) for idx in range(B)]
    X = torch.vstack(X)

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

    colordic, respdic = {}, {}
    for hkey in hkeys:
        Rcs = Rcs_all[hkey] = np.vstack(Rcs_all[hkey]) # value.shape = (6, #units, B)
        base = Rcs[0].std(axis=1) # shape = (#units,)
        color_indices = Rcs.std(axis=0).mean(axis=1)/base # shape = (#units,)
        colordic[hkey] = color_indices
        
        respdic[hkey] = []
        for unit in range(len(Rcs[0])):
            resp = responsive(Rcs[0][unit], abs=True) # it must be responsive to non-permuted images in abs value
            respdic[hkey].append(resp)
    return colordic, respdic

# get metrics
foi = get_foi()
ti, tiresp = get_TI()
ci, ciresp = get_CI()

top = 5
for hkey in hkeys[1:]: # ignore conv1
    idx_foi = man.nanargsort(foi[hkey]) # small to large, ignoring nan values
    idx_ti = man.idxargsort(np.array(ti[hkey]), np.where(tiresp[hkey])[0]) # sort only the responsive values
    idx_ci = man.idxargsort(np.array(ci[hkey]), np.where(ciresp[hkey])[0])

    nav.npsave(idx_foi[:5], nav.resultpath, f"responses_{mtype}", f"hkey={hkey}_unitsFoiMin.npy")
    nav.npsave(idx_foi[-5:], nav.resultpath, f"responses_{mtype}", f"hkey={hkey}_unitsFoiMax.npy")
    nav.npsave(idx_ti[:5], nav.resultpath, f"responses_{mtype}", f"hkey={hkey}_unitsTiMin.npy")
    nav.npsave(idx_ti[-5:], nav.resultpath, f"responses_{mtype}", f"hkey={hkey}_unitsTiMax.npy")
    nav.npsave(idx_ci[:5], nav.resultpath, f"responses_{mtype}", f"hkey={hkey}_unitsCiMin.npy")
    nav.npsave(idx_ci[-5:], nav.resultpath, f"responses_{mtype}", f"hkey={hkey}_unitsCiMax.npy")

