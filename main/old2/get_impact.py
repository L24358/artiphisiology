import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis

hkey = 3
units = range(192)

for unit in units:
    impact_dic = nav.pklload(nav.datapath, "results", "trace_AN", f"impact_hkey={hkey}_unit={unit}.pkl") # keys: preidx, values: impacts
    impact = np.array(list(impact_dic.values())) # shape = (#pre, top)
    sns.heatmap(impact)
    plt.xlabel("Rank of responses"); plt.ylabel("Pre-synaptic unit indices"); plt.suptitle(f"Layer {hkey}, Unit {unit}")
    vis.savefig(f"impact_hkey={hkey}_unit={unit}.png", folders=[nav.graphpath, "impact_AN"])