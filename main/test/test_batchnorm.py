import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import handytools.visualizer as vis
import spectools.basics as bcs
import spectools.models.models as mdl
from math import ceil, floor

def get_narrowing_stim(pxl, ival):
    center = (pxl-1)/2

    stims = []
    grid = range(1, pxl//2, ival)
    for stride in grid:
        lower = ceil(center - stride)
        upper = floor(center + stride + 1)
        
        stim = np.ones((pxl, pxl))
        stim[lower:upper, lower:upper] = np.zeros((upper-lower, upper-lower))
        stim = np.expand_dims(stim, axis=(0,1))
        stim = np.tile(stim, (1,3,1,1))
        stims.append(stim)
    return np.vstack(stims), grid

X, grid = get_narrowing_stim(200, 1);
X = torch.from_numpy(X.astype(np.float32))
bn = nn.BatchNorm2d(3)
Y1 = bn(X)
Y2 = bn(X[:10])

import pdb; pdb.set_trace()