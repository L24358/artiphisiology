"""
Feeding one single image suits deconvolved image most, not rotating image.
For rotating image, should use an average. This one uses average.

KILLED for no apparant reason.
"""

import gc
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
R_ori = nav.npload("/src", "results", f"responses_VGG16", f"key={hkey}_hollow=0_scale=1_light=1_lw=1_preproc=2.npy")
rot_info = nav.pklload("/src", "data", "stimulus", "shape_info.pkl")["rotation"]
rot_idx = nav.pklload("/src", "data", "stimulus", "shape_index.pkl")
image_array = nav.npload("/src", "data", "stimulus", f"stacked_rotated_hollow=0_lw=1_light=1_scale=1_preproc=2.npy")
image_array = torch.from_numpy(image_array)

# loop over previous layer (layer 8)
diff = []
for n in range(256): # CAUTION: Setting hkey does not adjust this!
    print("Working on unit ", n)
    model_copy = deepcopy(model)
    bcs.set_to_zero(model_copy, f"features.{hkey}.weight", unit, n)
    model_copy(image_array)
    R = model_copy.hidden_info[hkey][0]
    Rc = bcs.get_center_response(R)

    del model_copy

    R_modify = Rc[unit]
    diff.append(R_modify - R_ori[unit])
    nav.npsave(diff, "/src", "results", "subtraction_VGG16", f"mean_unit={unit}_key={hkey}_preunit={n}.npy")