"""
Follows trace_unit4. Important thing to note is that this uses the validation set.
"""

import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
import spectools.models.models as mdl

# hyperparameters
key = 11
unit = 435
N = 256
B = 31
top = 10
idx_tpe = "filt"

# load
idx = nav.npload(nav.datapath, "results", "subtraction_VGG16", f"{idx_tpe}idx.npy")

diffs = []
for i in range(B):
    diff = nav.npload(nav.datapath, "results", "subtraction_VGG16", f"imagenette_unit={unit}_key={key}_preunit={idx_tpe}{top}_B={i}.npy")
    diffs += list(diff)

print(np.mean(diffs))

