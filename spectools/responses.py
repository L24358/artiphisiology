import gc
import torch
import numpy as np
import handytools.visualizer as vis
import spectools.basics as bcs
import spectools.models.models as mdl
import handytools.navigator as nav

def get_response(hkeys, stim, folders, fname, mtype="AN", save=True):
    # params
    torch.cuda.empty_cache()
    device = "cpu"

    # load model
    if mtype == "AN": mfunc = mdl.get_alexnet
    elif mtype == "VGG16": mfunc = mdl.get_vgg16
    elif mtype == "VGG16b": mfunc = mdl.get_vgg16b
    elif mtype == "ResNet18": mfunc = mdl.get_resnet18
    model = mfunc(hidden_keys=hkeys).to(device)
    Rs = mdl.get_batch_hidden_info(model, stim)
    print(f"Using {mtype} on {device}.")

    # save results
    Rcs = {}
    for hkey in hkeys:
        Rc = bcs.get_center_response(Rs[hkey]) # shape = (#units,B)
        if save: nav.npsave(Rc, *folders, fname(hkey))
        Rcs[hkey] = Rc
    return Rcs

def get_response_wrapper(hkeys, stim, fname, mtype="AN", save=True, override=False):
    Rcs = {}
    folders = [nav.resultpath, f"responses_{mtype}"]

    mkeys = [] # missing keys
    for hkey in hkeys:
        if nav.exists(*folders, fname(hkey)):
            Rc = nav.npload(*folders, fname(hkey))
            Rcs[hkey] = Rc
        else:
            mkeys.append(hkey)

    print("Executing the follow missing keys: ", mkeys)
    if override:
        Rcs2 = get_response(hkeys, stim, folders, fname, mtype=mtype, save=save)
        Rcs.update(Rcs2)
    elif mkeys != []:
        Rcs2 = get_response(mkeys, stim, folders, fname, mtype=mtype, save=save)
        Rcs.update(Rcs2)
    return Rcs

def get_drr(hkeys, folders, fname, mtype="AN"):
    """
    Get dynamic range response.
    """
    torch.cuda.empty_cache()
    from spectools.stimulus.dataloader import Imagenette
    dataset = Imagenette("train")
    device = "cuda:0"
    folders = [nav.resultpath, f"responses_{mtype}"]

    # load model
    if mtype == "AN": mfunc = mdl.get_alexnet
    elif mtype == "VGG16": mfunc = mdl.get_vgg16
    elif mtype == "ResNet18": mfunc = mdl.get_resnet18

    Rcs = {}
    for hkey in hkeys:
        model = mfunc(hidden_keys=[hkey]).to(device)

        for i in range(0, len(dataset), 10):
            image, _, _ = dataset[i]
            image = image.unsqueeze(0)
            model(image.to(device))
            vis.print_batch(i, 1000)

        R = model.hidden_info[hkey] # is list, shape = (B,#units,h,w)
        R = np.vstack([r.cpu().detach().numpy() for r in R])
        Rc = bcs.get_center_response(R) # shape = (#units,B)
        nav.npsave(Rc, *folders, fname(hkey))
        Rcs[hkey] = Rc

        del model
        gc.collect()

    return Rcs

def get_drr_wrapper(hkeys, fname, mtype="AN"):
    Rcs = {}
    folders = [nav.resultpath, f"responses_{mtype}"]
    torch.cuda.empty_cache()

    mkeys = [] # missing keys
    for hkey in hkeys:
        if nav.exists(*folders, fname(hkey)):
            Rc = nav.npload(*folders, fname(hkey))
            Rcs[hkey] = Rc
        else:
            mkeys.append(hkey)

    Rcs2 = get_drr(mkeys, folders, fname, mtype=mtype)
    Rcs.update(Rcs2)
    return Rcs

if __name__ == "__main__":
    fname = lambda hkey: f"imgnettecr_hkey={hkey}.npy"
    get_drr_wrapper([0, 3], fname)