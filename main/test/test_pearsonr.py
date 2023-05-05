"""
Analyze degree of fill-outline invariance (FOI) of units in a particular layer of a particular network.
"""

import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
import handytools.manipulator as man
from scipy.stats import pearsonr
from spectools.metrics.metrics import MSE, R2

# argv
argv_dic = man.argv_to_dic(sys.argv)

# hyperparameters
mtype = man.argv_manager(argv_dic, 1, "AN")
hidden_key = man.argv_manager(argv_dic, 2, 3, tpe=int)
scale = 1.0
light = True
linewidth = 1
preprocess = man.argv_manager(argv_dic, 3, 2, tpe=int)
print(f"Begin processing: Fill-outline invariance analysis, for network={mtype}, key={hidden_key}, preprocess={preprocess}.")

# load data
R_fill = nav.npload(nav.homepath, "results", f"responses_{mtype}", f"key={hidden_key}_hollow=0_scale={scale}_light={int(light)}_lw={linewidth}_preproc={preprocess}.npy")
R_holl = nav.npload(nav.homepath, "results", f"responses_{mtype}", f"key={hidden_key}_hollow=1_scale={scale}_light={int(light)}_lw={linewidth}_preproc={preprocess}.npy")

# analysis
dic = {}
for s in range(len(R_fill)):
    pr = pearsonr(R_fill[s], R_holl[s])[0]
    pr2 = np.corrcoef(R_fill[s], R_holl[s])[0][1]

    if round(pr,2) != round(pr2,2):
        print(round(pr,2), round(pr2,2))