import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis

hidden_key = 8
linewidth = 1
data_type = "rotated"

fh_dic = nav.pklload("/src", "data", f"fillholl_{data_type}_key={hidden_key}_lw={linewidth}", "fit_metrics.pkl")
oo_dic = nav.pklload("/src", "data", f"onoff_{data_type}_key={hidden_key}_lw={linewidth}", "fit_metrics.pkl")

keys = list(fh_dic.keys() & oo_dic.keys()) # TODO: filter by response sparsity
r_fh = [fh_dic[s][0] for s in keys] # TODO: get rid of the other metric
r_oo = [oo_dic[s] for s in keys]

def plot_45degree(ax, x, y): # TODO: move into handytools
    minn = min([min(x), min(y)])
    maxx = max([max(x), max(y)])
    ax.plot([minn, maxx], [minn, maxx], "k--") # plot x=y
    return ax

ax = vis.simpleaxis(None) # TODO: use None as default
ax.scatter(r_fh, r_oo, color="b")
# ax = plot_45degree(ax, r_fh, r_oo) # TODO: fix
ax.set_xlabel("fill-outline invariance pear. corr.")
ax.set_ylabel("on-off invariance pear. corr.")
vis.savefig()

from scipy.stats import pearsonr
print(pearsonr(r_fh, r_oo)[0])