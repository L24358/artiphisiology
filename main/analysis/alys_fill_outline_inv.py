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
scale = 1
light = True
linewidth = 1
preprocess = man.argv_manager(argv_dic, 3, 2, tpe=int)
print(f"Begin processing: Fill-outline invariance analysis, for network={mtype}, key={hidden_key}, preprocess={preprocess}.")

# load data
R_fill = nav.npload(nav.homepath, "results", f"responses_{mtype}", f"key={hidden_key}_hollow=0_scale={scale}_light={int(light)}_lw={linewidth}_preproc={preprocess}.npy")
R_holl = nav.npload(nav.homepath, "results", f"responses_{mtype}", f"key={hidden_key}_hollow=1_scale={scale}_light={int(light)}_lw={linewidth}_preproc={preprocess}.npy")
folders = [f"FOI_{mtype}", f"key={hidden_key}_preproc={preprocess}"]

# catch warnings
warnings.filterwarnings("ignore")

# analysis
dic = {}
for s in range(len(R_fill)):
    pr = pearsonr(R_fill[s], R_holl[s])[0]
    if not np.isnan(pr): # if this is nan, implies that std(x)=0 or std(y)=0, and it is meaningless to plot
        mse = MSE(R_fill[s], R_holl[s])
        r2 = R2(R_fill[s], R_holl[s])

        plt.scatter(R_fill[s], R_holl[s], color="b")
        minn = min([min(R_fill[s]), min(R_holl[s])])
        maxx = max([max(R_fill[s]), max(R_holl[s])])
        plt.plot([minn, maxx], [minn, maxx], "k--") # plot x=y
        plt.xlabel("Resp. to filled shapes"); plt.ylabel("Resp. to outlines"); plt.title(f"$r^2$: {round(r2,2)}, \u03C1: {round(pr, 2)}") 
        vis.savefig(f"idx={s}.png", folders=folders)
        dic[s] = [pr, mse, r2]
nav.pklsave(dic, nav.homepath, "results", *folders, "fit_metrics.pkl")