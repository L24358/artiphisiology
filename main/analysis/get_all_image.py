import sys
import torch
import numpy as np
import spectools.basics as bcs
import spectools.models.models as mdl
import handytools.navigator as nav
import handytools.manipulator as man

hollow = 0
linewidth = 1
preprocess = 2
light = 1
scale = 1

rot_info = nav.pklload("/src", "data", "stimulus", "shape_info.pkl")["rotation"]

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

nav.npsave(image_arrays, "/src", "data", "stimulus", f"stacked_rotated_hollow={int(hollow)}_lw={linewidth}_light={int(light)}_scale={scale}_preproc={preprocess}.npy")