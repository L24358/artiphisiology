"""
Get responses of networks to different images (light, scale, hollow, etc.).

@ Updates:
    - Added compatibility with VGG16.
"""

import sys
import torch
import numpy as np
import spectools.basics as bcs
import spectools.models.models as mdl
import handytools.navigator as nav
import handytools.manipulator as man

# argv
argv_dic = man.argv_to_dic(sys.argv)

# hyperparameters
mtype = man.argv_manager(argv_dic, 1, "VGG16")
hidden_key = man.argv_manager(argv_dic, 2, 22, tpe=int)
hollow = man.argv_manager(argv_dic, 3, True, tpe=man.bool_int)
scale = man.argv_manager(argv_dic, 4, 1, tpe=int)
light = man.argv_manager(argv_dic, 5, True, tpe=man.bool_int)
linewidth = man.argv_manager(argv_dic, 6, 1, tpe=int)
foldername = f"_rotated_hollow={int(hollow)}_lw={linewidth}"

# load model
if mtype == "AN": mfunc = mdl.get_alexnet
elif mtype == "VGG16": mfunc = mdl.get_vgg16
model = mfunc(hidden_keys=[hidden_key])
rot_info = nav.pklload(nav.datapath, "stimulus", "shape_info.pkl")["rotation"]

image_arrays = []
for s in range(51):
    for r in range(rot_info[s]):
        image_array = nav.npload(nav.datapath, "stimulus" + foldername, f"idx={s}_pxl=227_r={r}.npy") # shape = (227, 227, 4)

        # preprocess image value
        image_array = image_array/255. # dark BG (0), light image (1)
        if not light: image_array *= -1 # light BG (0), dark image (-1)
        image_array *= scale

        # preprocess image dimension
        image_array = np.swapaxes(image_array, 0, -1)
        image_array = np.swapaxes(image_array, 1, 2)[:3, :, :] # shape = (3, 227, 227)
        image_array = np.expand_dims(image_array, 0)
        image_arrays.append(image_array)
image_arrays = np.vstack(image_arrays)
print("Image preprocessed.")

X = torch.from_numpy(image_arrays)
model(X)
print("Model output completed.")
R = model.hidden_info[hidden_key][0] # shape = (51, 256, 13, 13)
Rc = bcs.get_center_response(R) # shape = (256, 51)
nav.npsave(Rc, nav.homepath, "results", "responses"+foldername+f"_light={int(light)}_scale={scale}", f"{mtype}_CR_stim=shape_key={hidden_key}.npy")