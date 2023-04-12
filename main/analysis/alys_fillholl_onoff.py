import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
from scipy.stats import pearsonr

hidden_key = 8
linewidth = 1
data_type = "rotated"
mtype = "AN"

fh_dic = nav.pklload("/src", "data", f"{mtype}_fillholl_{data_type}_key={hidden_key}_lw={linewidth}", "fit_metrics.pkl")
oo_dic = nav.pklload("/src", "data", f"{mtype}_onoff_{data_type}_key={hidden_key}_lw={linewidth}", "fit_metrics.pkl")

keys = list(fh_dic.keys() & oo_dic.keys())
r_fh = np.array([fh_dic[s] for s in keys]).flatten()
r_oo = np.array([oo_dic[s] for s in keys]).flatten()

ax = vis.simpleaxis()
ax.scatter(r_fh, r_oo, color="b")
ax = vis.plot_45degree(ax, r_fh, r_oo)
ax.set_xlabel("fill-outline invariance pear. corr.")
ax.set_ylabel("on-off invariance pear. corr.")
vis.savefig()

print(pearsonr(r_fh, r_oo)[0])
