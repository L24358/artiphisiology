import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis

hkey = 3
units = range(192)

vals = []
for unit in units:
    impact_dic = nav.pklload(nav.datapath, "results", "trace_AN", f"impact_hkey={hkey}_unit={unit}.pkl") # keys: preidx, values: impacts
    impact = np.array(list(impact_dic.values())).T # shape = (top, #pre)
    C = np.corrcoef(impact)
    val = (C.sum() - 20)/380
    print(f"For Unit {unit}, VAL: {val}")
    vals.append(val)

print(np.argsort(vals))
nav.npsave(vals, nav.datapath, "temp", "corr_spread.npy")