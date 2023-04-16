"""
@ Updates:
    - Include option to use image preprocessing step s.t. it conforms with torch hub documentation
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
preprocess = man.argv_manager(argv_dic, 7, 2, tpe=int)
print(f"Begin processing: network={mtype}, key={hidden_key}, hollow={bool(hollow)}, scale={scale}, light={bool(light)}, linewidth={linewidth}, preprocess={preprocess}")

# load model, info and define parameters
if mtype == "AN": mfunc = mdl.get_alexnet
elif mtype == "VGG16": mfunc = mdl.get_vgg16
model = mfunc(hidden_keys=[hidden_key])
rot_info = nav.pklload("/src", "data", "stimulus", "shape_info.pkl")["rotation"]

# process image
image_arrays = []
for s in range(51): # there are 51 base shapes
    for r in range(rot_info[s]):
        image_array = nav.npload("/src", "data", f"stimulus_rotated_hollow={int(hollow)}_lw={linewidth}", f"idx={s}_pxl=227_r={r}.npy") # shape = (227, 227, 4)

        # preprocess image value
        preproc_dic = {1: bcs.preprocess1, 2: bcs.preprocess2}
        image_array = preproc_dic[preprocess](image_array, light, scale)

        # preprocess image dimension
        image_array = np.swapaxes(image_array, 0, -1)
        image_array = np.swapaxes(image_array, 1, 2)[:3, :, :] # shape = (3, 227, 227)
        image_array = np.expand_dims(image_array, 0)
        image_arrays.append(image_array)
image_arrays = np.vstack(image_arrays)

X = torch.from_numpy(image_arrays)
model(X)
R = model.hidden_info[hidden_key][0] # shape = (51, 256, 13, 13)
Rc = bcs.get_center_response(R) # shape = (256, 51)
nav.npsave(Rc, "/src", "results", 
           f"responses_{mtype}",
           f"key={hidden_key}_hollow={int(hollow)}_scale={scale}_light={int(light)}_lw={linewidth}_preproc={preprocess}.npy")