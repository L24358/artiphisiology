"""
Double chek that the layer indices of ResNet18, by block.
"""

import torch
import numpy as np
import spectools.models.models as mdl

mtype = "ResNet18"
device = "cuda:0"
X = np.random.normal(0, 1, size=(1,3,227,227))
X = torch.from_numpy(X).to(device)

for hkey in [0, 4, 7, 9, 12, 14, 17, 20, 23, 25, 28, 31, 34, 36, 39, 42, 45]:
    hkeys = [hkey]
    model = mdl.load_model(mtype, hkeys, device)
    model(X, by_block=False)