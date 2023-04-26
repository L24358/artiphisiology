"""
Store responses to Imagenette for modulated network.

@ TODO:
    - Not finished.
"""

import gc
import torch
import handytools.navigator as nav
import spectools.basics as bcs
from copy import deepcopy
from torch.utils.data import DataLoader
from spectools.models.models import get_vgg16
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
key = 11
unit = 435 # this is the target unit in layer ``key``
N = 258
bs = 128
device = "cuda:0"
torch.manual_seed(42)
print(f"CAUTION: the pre-layer number of units ``N`` needs to be adjusted manually based on key. Current key is {key}, and N is {N}.")

# define dataset, models
dataset = Imagenette()
train_dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
model = get_vgg16(hidden_keys=[key]).to(device)

start, end = 0, 100 # which batch to start and end with
model.eval()
for i, data in enumerate(train_dataloader):

    print("Batch ", i)
    if i >= start:

        image, label, img_idx = data

        for n in range(N):
            pass

            print("Turning off unit ", n)
            model_copy = deepcopy(model)
            model_copy.eval()
            bcs.set_to_zero(model_copy, f"features.{key}.weight", unit, n) # turns off n --> unit connections

            model_copy(image.to(device), premature_quit = True)
            R = model_copy.hidden_info[key][0]
            Rc = bcs.get_center_response(R)
            R_modify = Rc[unit] # save R_modify instead of diference
            nav.npsave(R_modify, nav.datapath, "results", "leaveoneout_VGG16", f"imagenette_unit={unit}_key={key}_preunit={n}.npy")

            del model_copy
            del image, label, img_idx
            model.reset_storage()
            gc.collect()

    if i > end: break



