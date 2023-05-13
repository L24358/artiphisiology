"""
Get MEI (most exciting image) and obtain leave-one-out responses from the network.

@ TODO:
    - Feeding one single image suits deconvolved image most, not rotating image. For rotating image, should use an average.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
import spectools.basics as bcs
import spectools.models.models as mdl
from copy import deepcopy

# unit of interest
unit = 435
hkey = 11 # CAUTION: Setting hkey does not adjust the previous layer nunit!

# load data
model = mdl.get_vgg16(hidden_keys=[11])
R_ori = nav.npload(nav.homepath, "results", f"responses_VGG16", f"key={hkey}_hollow=0_scale=1_light=1_lw=1_preproc=2.npy")
rot_info = nav.pklload(nav.datapath, "stimulus", "shape_info.pkl")["rotation"]
rot_idx = nav.pklload(nav.datapath, "stimulus", "shape_index.pkl")

# get most responsive image
mag = list(abs(R_ori[unit])) # largest response of the unit
img_idx = mag.index(max(mag)) # index of the image
s, r = rot_idx[img_idx]
image_array = nav.npload(nav.datapath, f"stimulus_rotated_hollow=0_lw=1", f"idx={s}_pxl=227_r={r}.npy")
image_array = bcs.preprocess2(image_array, 1, 1)
image_array = bcs.preprocess3(image_array)
image_array = torch.from_numpy(image_array)

# loop over previous layer (layer 8)
diff = []
for n in range(256): # CAUTION: Setting hkey does not adjust this!
    model_copy = deepcopy(model)
    bcs.set_to_zero(model_copy, f"features.{hkey}.weight", unit, n)
    model_copy(image_array)
    R = model_copy.hidden_info[hkey][0]
    Rc = bcs.get_center_response(R)

    del model_copy
    R_modify = Rc[unit].item()
    diff.append(R_modify - R_ori[unit, img_idx])
nav.npsave(np.array(diff), nav.homepath, "results", "subtraction_VGG16", f"max_unit={unit}_key={hkey}.npy")