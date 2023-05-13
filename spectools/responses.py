import spectools.basics as bcs
import spectools.models.models as mdl
import handytools.navigator as nav

def get_response(hkeys, stim, folders, fname, mtype="AN"):
    # params
    device = "cuda:0"

    # load model
    if mtype == "AN": mfunc = mdl.get_alexnet
    elif mtype == "VGG16": mfunc = mdl.get_vgg16
    elif mtype == "ResNet18": mfunc = mdl.get_resnet18
    model = mfunc(hidden_keys=hkeys).to(device)
    model(stim.to(device))

    # save results
    Rcs = {}
    for hkey in hkeys:
        R = model.hidden_info[hkey][0].cpu() # shape = (B,#units,h,w)
        Rc = bcs.get_center_response(R) # shape = (#units,B)
        nav.npsave(Rc, *folders, fname(hkey))
        Rcs[hkey] = Rc
    return Rcs

def get_response_wrapper(hkeys, stim, fname, mtype="AN"):
    Rcs = {}
    folders = [nav.resultpath, f"responses_{mtype}"]

    mkeys = [] # missing keys
    for hkey in hkeys:
        if nav.exists(*folders, fname(hkey)):
            Rc = nav.npload(*folders, fname(hkey))
            Rcs[hkey] = Rc
        else:
            mkeys.append(hkey)

    Rcs2 = get_response(mkeys, stim, folders, fname, mtype="AN")
    Rcs.update(Rcs2)
    return Rcs
