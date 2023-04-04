import torch
import numpy as np
import spectools.basics as bcs
import spectools.models.models as mdl
import handytools.navigator as nav

# hyperparameters
hidden_key = 10

# load model
model = mdl.get_alexnet(hidden_keys=[hidden_key])

image_arrays = []
for s in range(51):
    image_array = nav.npload("/src", "data", "stimulus", f"idx={s}_pxl=227.npy") # shape = (227, 227, 4)
    image_array = np.swapaxes(image_array, 0, -1)
    image_array = np.swapaxes(image_array, 1, 2)[:3, :, :] # shape = (3, 227, 227)
    image_array = np.expand_dims(image_array, 0)
    image_arrays.append(image_array)
image_arrays = np.vstack(image_arrays)

X = torch.from_numpy(image_arrays)
model(X)
R = model.hidden_info[hidden_key][0] # shape = (51, 256, 13, 13)
Rc = bcs.get_center_response(R)
nav.npsave(Rc, "/src", "data", "responses", f"CR_stim=shape_key={hidden_key}.npy")
