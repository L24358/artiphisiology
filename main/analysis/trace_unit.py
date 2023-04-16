import torch
import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
import spectools.basics as bcs
import spectools.models.models as mdl
from copy import deepcopy
from handytools.catcher import InputError

# unit of interest
unit = 435

# load data
model = mdl.get_vgg16(hidden_keys=[11])
R_ori = nav.npload("/src", "results", f"responses_VGG16", f"key=8_hollow=0_scale=1_light=1_lw=1_preproc=2.npy")
rot_info = nav.pklload("/src", "data", "stimulus", "shape_info.pkl")["rotation"]

# set to zero function
def set_to_zero(model, target, id1, id2):
    flag = True
    for name, param in model.named_parameters():
        if name == target:
            param_clone = param.clone()
            param_clone[id1][id2] = torch.zeros(param_clone[id1][id2].shape)
            param.data = param_clone
            flag = False
    if flag: raise InputError(f"{target} is not found in model.named_parameters().")

# get most responsive image
mag = list(abs(R_ori[unit]))
img_idx = mag.index(max(mag))
# process image
count = 0
for s in range(51): # there are 51 base shapes
    flag = False
    for r in range(rot_info[s]):
        if count == img_idx:
            image_array = nav.npload("/src", "data", f"stimulus_rotated_hollow=0_lw=1", f"idx={s}_pxl=227_r={r}.npy") # shape = (227, 227, 4)
            flag = True
            break
    if flag: break
# preprocess image value and dimension
image_array = bcs.preprocess2(image_array, 1, 1)
image_array = np.swapaxes(image_array, 0, -1)
image_array = np.swapaxes(image_array, 1, 2)[:3, :, :] # shape = (3, 227, 227)
image_array = np.expand_dims(image_array, 0)

# loop over previous layer (layer 8)
diff = []
for n in range(256):
    model_copy = deepcopy(model)
    set_to_zero(model_copy, "features.11.weight", unit, n)
    model_copy(image_array)
    R = model.hidden_info[11][0] # shape = (# image, # unit, 13, 13)
    Rc = bcs.get_center_response(R) # shape = (# unit, # image)

    del model_copy
    R_modify = Rc[unit].squeeze()
    diff.append(R_modify - R_ori[unit])

plt.plot(diff)
vis.savefig()