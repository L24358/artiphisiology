import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

hidden_key = 8
light = True
linewidth = 1
data_type = "rotated"

R_fill = nav.npload("/src", "data", f"responses_{data_type}_hollow=0_lw={linewidth}_light={int(light)}", f"CR_stim=shape_key={hidden_key}.npy")
R_holl = nav.npload("/src", "data", f"responses_{data_type}_hollow=1_lw={linewidth}_light={int(light)}", f"CR_stim=shape_key={hidden_key}.npy")
folders = [f"fillholl_{data_type}_key={hidden_key}_lw={linewidth}"]

dic = {}
for s in range(51):
    pr = pearsonr(R_fill[s], R_holl[s])[0]
    if not np.isnan(pr):
        r2 = r2_score(R_fill[s], R_holl[s])
        plt.scatter(R_fill[s], R_holl[s], color="k")
        plt.xlabel("Resp. to filled shapes"); plt.ylabel("Resp. to outlines"); plt.title(f"\u03C1: {round(pr, 2)}, $r^2$: {round(r2, 2)}")
        vis.savefig(f"idx={s}.png", folders=folders)
        dic[s] = [r2, pr]
nav.pklsave(dic, "/src", "data", folders[0], "r2_pr.pkl")