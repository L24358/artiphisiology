"""
Analyze the degree of Dynamic-Range Invariance (DRI) of a particular layer of a particular network.
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
hidden_key = man.argv_manager(argv_dic, 2, 8, tpe=int)
hollow = False
linewidth = 1
light = True
scale = man.argv_manager(argv_dic, 4, 2, tpe=float)
preprocess = man.argv_manager(argv_dic, 3, 2, tpe=int)
print(f"Begin processing: Fill-outline invariance analysis, for network={mtype}, key={hidden_key}, preprocess={preprocess}, scale={scale}.")

# catch warnings
warnings.filterwarnings("ignore")

# load data
R_baseline = nav.npload(nav.homepath, "results", f"responses_{mtype}", f"key={hidden_key}_hollow={int(hollow)}_scale=1.0_light={int(light)}_lw={linewidth}_preproc={preprocess}.npy")
R_scaled = nav.npload(nav.homepath, "results", f"responses_{mtype}", f"key={hidden_key}_hollow={int(hollow)}_scale={scale}_light={int(light)}_lw={linewidth}_preproc={preprocess}.npy")
folders = [f"DRI_{mtype}_scale={scale}", f"key={hidden_key}_preproc={preprocess}"]

# analysis
dic = {}
for s in range(len(R_baseline)):
    pr = pearsonr(R_baseline[s], R_scaled[s])[0]
    if not np.isnan(pr):
        mse = MSE(R_baseline[s], R_scaled[s])
        r2 = R2(R_baseline[s], R_scaled[s])

        plt.scatter(R_baseline[s], R_scaled[s], color="b")
        minn = min([min(R_baseline[s]), min(R_scaled[s])])
        maxx = max([max(R_baseline[s]), max(R_scaled[s])])
        plt.plot([minn, maxx], [minn, maxx], "k--") # plot x=y
        plt.xlabel("Resp. to baseline stimuli"); plt.ylabel(f"Resp. to scaled (s={scale}) stimuli"); plt.title(f"$r^2$: {round(r2,2)}, \u03C1: {round(pr, 2)}") 
        vis.savefig(f"idx={s}.png", folders=folders)
        dic[s] = [pr, mse, r2]
nav.pklsave(dic, nav.homepath, "results", *folders, "fit_metrics.pkl")