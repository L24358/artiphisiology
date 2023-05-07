import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis

hkey = 3
units = range(192)

def normalize(arr):
    rep_mean = arr.mean(axis=0) # shape = # pre units
    arr -= arr.mean()
    arr /= rep_mean.std()
    return arr

for unit in units:
    impact_dic = nav.pklload(nav.datapath, "results", "trace_AN", f"impact_hkey={hkey}_unit={unit}.pkl") # keys: preidx, values: impacts
    impact = np.array(list(impact_dic.values())).T # shape = (top, #pre)
    impact = normalize(impact)
    C = np.cov(impact)
    print(f"For Unit {unit}, MEAN: {C.mean()}, STD: {C.std()}")
    # if unit == 177:
    #     plt.hist(C.flatten())
    #     vis.savefig()