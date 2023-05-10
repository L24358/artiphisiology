"""
Get response to TK texture stimulus.
"""

import sys
import torch
import numpy as np
import spectools.basics as bcs
import spectools.models.models as mdl
import handytools.navigator as nav
import handytools.manipulator as man
from copy import deepcopy

# argv
argv_dic = man.argv_to_dic(sys.argv)

# hyperparameters
mtype = man.argv_manager(argv_dic, 1, "AN")
hidden_key = man.argv_manager(argv_dic, 2, 3, tpe=int)
device = "cuda:0"

# load model, info and define parameters
if mtype == "AN": mfunc = mdl.get_alexnet
elif mtype == "VGG16": mfunc = mdl.get_vgg16
elif mtype == "ResNet18": mfunc = mdl.get_resnet18
model = mfunc(hidden_keys=[hidden_key]).to(device)
rot_info = nav.pklload(nav.datapath, "stimulus", "shape_info.pkl")["rotation"]

# process image
image_arrays = []
color_arrays = []
combo = [[0], [1], [2], [0,1], [0,2], [1,2]]
for i in range(225, 393):
    image_array = nav.npload(nav.homepath, "data", "stimulus_TK", f"idx={i}_pxl=227.npy")
    image_array = np.expand_dims(image_array, (0,1))
    image_array = np.tile(image_array, (1,3,1,1))/255.
    image_arrays.append(image_array)

    for c in combo:
        color_array = deepcopy(image_array)
        color_array[:,c,...] = np.zeros((1,len(c),227,227))
        color_arrays.append(color_array)
image_arrays = np.vstack(image_arrays)
color_arrays = np.vstack(color_arrays)

X = torch.from_numpy(image_arrays).to(device)
model(X)
R = model.hidden_info[hidden_key][0] # shape = (51, 256, 13, 13)
Rc = bcs.get_center_response(R.cpu()) # shape = (256, 51)
nav.npsave(Rc, nav.homepath, "results", 
           f"responses_{mtype}",
           f"TKtexture_hkey={hidden_key}.npy")

X = torch.from_numpy(color_arrays).to(device)
model(X)
R = model.hidden_info[hidden_key][1] # shape = (51, 256, 13, 13)
Rc = bcs.get_center_response(R.cpu()) # shape = (256, 51)
nav.npsave(Rc, nav.homepath, "results", 
           f"responses_{mtype}",
           f"TKtexturecolor_hkey={hidden_key}.npy")