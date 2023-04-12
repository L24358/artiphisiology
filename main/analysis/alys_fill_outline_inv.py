import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
from scipy.stats import pearsonr
from spectools.metrics.metrics import NMSE

hidden_key = 8
light = True
linewidth = 1
data_type = "rotated"
scale = 1
mtype = "VGG16"

R_fill = nav.npload("/src", "data", f"responses_{data_type}_hollow=0_lw={linewidth}_light={int(light)}_scale={scale}", f"{mtype}_CR_stim=shape_key={hidden_key}.npy")
R_holl = nav.npload("/src", "data", f"responses_{data_type}_hollow=1_lw={linewidth}_light={int(light)}_scale={scale}", f"{mtype}_CR_stim=shape_key={hidden_key}.npy")
folders = [f"{mtype}_fillholl_{data_type}_key={hidden_key}_lw={linewidth}"]

dic = {}
for s in range(len(R_fill)):
    pr = pearsonr(R_fill[s], R_holl[s])[0]
    if not np.isnan(pr):
        plt.scatter(R_fill[s], R_holl[s], color="b")
        minn = min([min(R_fill[s]), min(R_holl[s])])
        maxx = max([max(R_fill[s]), max(R_holl[s])])
        plt.plot([minn, maxx], [minn, maxx], "k--") # plot x=y
        plt.xlabel("Resp. to filled shapes"); plt.ylabel("Resp. to outlines"); plt.title(f"\u03C1: {round(pr, 2)}") 
        vis.savefig(f"idx={s}.png", folders=folders)
        dic[s] = [pr]
nav.pklsave(dic, "/src", "data", folders[0], "fit_metrics.pkl")