"""
Double chek that the FC layer indices are correct for AN, VGG11 (VGG16), VGG16 (VGG16b).
"""

import torch
import numpy as np
import spectools.models.models as mdl

mtype = "VGG16"
device = "cuda:0"
X = np.random.normal(0, 1, size=(1,3,227,227))
X = torch.from_numpy(X).to(device)

for hkey in [22, 25, 28]:
    hkeys = [hkey]
    model = mdl.load_model(mtype, hkeys, device)
    model(X)