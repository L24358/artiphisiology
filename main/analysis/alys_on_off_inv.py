import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
from scipy.stats import pearsonr
from spectools.metrics.metrics import NMSE

hidden_key = 8
hollow = False
linewidth = 1
data_type = "rotated"
scale = 1
mtype = "AN"

R_light = nav.npload("/src", "data", f"responses_{data_type}_hollow={int(hollow)}_lw={linewidth}_light=1_scale={scale}", f"{mtype}_CR_stim=shape_key={hidden_key}.npy")
R_dark = nav.npload("/src", "data", f"responses_{data_type}_hollow={int(hollow)}_lw={linewidth}_light=0_scale={scale}", f"{mtype}_CR_stim=shape_key={hidden_key}.npy")
folders = [f"{mtype}_onoff_{data_type}_key={hidden_key}_lw={linewidth}"]

dic = {}
for s in range(len(R_light)):
    pr = pearsonr(R_light[s], R_dark[s])[0]
    if not np.isnan(pr):
        plt.scatter(R_light[s], R_dark[s], color="b")
        minn = min([min(R_light[s]), min(R_dark[s])])
        maxx = max([max(R_light[s]), max(R_dark[s])])
        plt.plot([minn, maxx], [minn, maxx], "k--") # plot x=y
        plt.xlabel("Resp. to light stimuli"); plt.ylabel("Resp. to dark stimuli"); plt.title(f"\u03C1: {round(pr, 2)}") 
        vis.savefig(f"idx={s}.png", folders=folders)
        dic[s] = [pr]
nav.pklsave(dic, "/src", "data", folders[0], "fit_metrics.pkl")