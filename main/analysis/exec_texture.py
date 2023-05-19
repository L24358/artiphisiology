"""
Response to TK texture stimulus.

@ Notes:
    - A copy of get_response5.py.
"""

import sys
import torch
import numpy as np
import spectools.basics as bcs
import spectools.models.models as mdl
import handytools.navigator as nav
import handytools.manipulator as man
from spectools.responses import get_response_wrapper

# hyperparameters
mtype = "ResNet18"
hkeys = list(mdl.ResNet18_layer.keys())
device = "cpu"

# load model, info and define parameters
if mtype == "AN": mfunc = mdl.get_alexnet
elif mtype == "VGG16b": mfunc = mdl.get_vgg16b
elif mtype == "ResNet18": mfunc = mdl.get_resnet18
model = mfunc(hidden_keys=hkeys).to(device)

# process image
image_arrays = []
for i in range(393): # 225 to 393 is texture
    image_array = nav.npload(nav.homepath, "data", "stimulus_TK", f"idx={i}_pxl=227.npy")
    image_array = np.expand_dims(image_array, (0,1))

    temp = image_array.flatten()
    print("Index: ", i, ", MIN: ", min(temp),  ", MAX: ", max(temp))

    # TODO: consider rescaling
    image_array = np.tile(image_array, (1,3,1,1))/255.
    image_arrays.append(image_array)
image_arrays = np.vstack(image_arrays)

fname = lambda hkey: f"hkey={hkey}_TK.npy"
Rs = get_response_wrapper(hkeys, torch.from_numpy(image_arrays), fname, mtype=mtype, save=False, override=True)

# save results
for hkey in hkeys:
    Rc = bcs.get_center_response(Rs[hkey].T) # shape = (256, 51)
    nav.npsave(Rc, nav.resultpath, f"responses_{mtype}", f"hkey={hkey}_TKRc.npy")
