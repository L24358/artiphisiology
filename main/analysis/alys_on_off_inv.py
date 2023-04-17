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
scale = 1
preprocess = man.argv_manager(argv_dic, 3, 2, tpe=int)
print(f"Begin processing: Fill-outline invariance analysis, for network={mtype}, key={hidden_key}, preprocess={preprocess}.")


# catch warnings
warnings.filterwarnings("ignore")

# load data
R_light = nav.npload("/src", "results", f"responses_{mtype}", f"key={hidden_key}_hollow=0_scale={scale}_light=1_lw={linewidth}_preproc={preprocess}.npy")
R_dark = nav.npload("/src", "results", "/src", "results", f"responses_{mtype}", f"key={hidden_key}_hollow=0_scale={scale}_light=0_lw={linewidth}_preproc={preprocess}.npy")
folders = [f"OOI_{mtype}", f"key={hidden_key}_preproc={preprocess}"]

# analysis
dic = {}
for s in range(len(R_light)):
    pr = pearsonr(R_light[s], R_dark[s])[0]
    if not np.isnan(pr):
        mse = MSE(R_light[s], R_dark[s])
        r2 = R2(R_light[s], R_dark[s])

        plt.scatter(R_light[s], R_dark[s], color="b")
        minn = min([min(R_light[s]), min(R_dark[s])])
        maxx = max([max(R_light[s]), max(R_dark[s])])
        plt.plot([minn, maxx], [minn, maxx], "k--") # plot x=y
        plt.xlabel("Resp. to light stimuli"); plt.ylabel("Resp. to dark stimuli"); plt.title(f"$r^2$: {round(r2,2)}, \u03C1: {round(pr, 2)}") 
        vis.savefig(f"idx={s}.png", folders=folders)
        dic[s] = [pr, mse, r2]
nav.pklsave(dic, "/src", "results", *folders, "fit_metrics.pkl")