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

mtype = man.argv_manager(argv_dic, 1, "VGG16")
hidden_key = man.argv_manager(argv_dic, 2, 8, tpe=int)
scale = 1
light = True
linewidth = 1
data_type = "rotated"

R_fill = nav.npload("/src", "results", f"responses_{data_type}_hollow=0_lw={linewidth}_light={int(light)}_scale={scale}", f"{mtype}_CR_stim=shape_key={hidden_key}.npy")
R_holl = nav.npload("/src", "results", f"responses_{data_type}_hollow=1_lw={linewidth}_light={int(light)}_scale={scale}", f"{mtype}_CR_stim=shape_key={hidden_key}.npy")
folders = [f"{mtype}_fillholl_{data_type}_key={hidden_key}_lw={linewidth}"]

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
nav.pklsave(dic, "/src", "results", folders[0], "fit_metrics.pkl")