import sys
import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
import handytools.manipulator as man
from scipy.stats import pearsonr
from spectools.metrics.metrics import MSE, R2

# argv
argv_dic = man.argv_to_dic(sys.argv)

mtype = man.argv_manager(argv_dic, 1, "AN")
hidden_key = man.argv_manager(argv_dic, 2, 8, tpe=int)
hollow = False
linewidth = 1
data_type = "rotated"
scale = 1

R_light = nav.npload("/src", "results", f"responses_{data_type}_hollow={int(hollow)}_lw={linewidth}_light=1_scale={scale}", f"{mtype}_CR_stim=shape_key={hidden_key}.npy")
R_dark = nav.npload("/src", "results", f"responses_{data_type}_hollow={int(hollow)}_lw={linewidth}_light=0_scale={scale}", f"{mtype}_CR_stim=shape_key={hidden_key}.npy")
folders = [f"{mtype}_onoff_{data_type}_key={hidden_key}_lw={linewidth}"]

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
        plt.xlabel("Resp. to light stimuli"); plt.ylabel("Resp. to dark stimuli"); plt.title(f"\u03C1: {round(pr, 2)}") 
        vis.savefig(f"idx={s}.png", folders=folders)
        dic[s] = [pr, mse, r2]
nav.pklsave(dic, "/src", "results", folders[0], "fit_metrics.pkl")