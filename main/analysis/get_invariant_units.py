import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
import handytools.manipulator as man
import spectools.basics as bcs

# hyperparameters
preprocess = 2
thre = 0.8
hk = 3
top = 20

# helpful functions
def get_dict(keys, values):
    """Create dictionary based on keys, values."""
    dic = {}
    for i, key in enumerate(keys): dic[key] = values[i]
    return dic

def get_r2(datatype, mtype, key, preprocess):
    """Get filtered r-squared values (i.e. ReLU(r2))."""
    metrics = nav.pklload(nav.homepath, "results", f"{datatype}_{mtype}", f"key={key}_preproc={preprocess}", "fit_metrics.pkl")
    metrics_array = np.asarray(list(metrics.values())) # shape = (# responsive units, # metrics)
    r2_filtered = np.maximum(metrics_array[:,2], 0) # ReLU(r^2)
    return r2_filtered, list(metrics.keys()), get_dict(list(metrics.keys()), r2_filtered)

r2_foi, idx_foi, foi_dic = get_r2("FOI", "AN", hk, preprocess)
r2_ooi, idx_ooi, ooi_dic = get_r2("OOI", "AN", hk, preprocess)
r2_dri, idx_dri, dri_dic = get_r2("DRI", "AN_scale=0.5", hk, preprocess)
new_dic = man.combine_dict(foi_dic) # , ooi_dic, dri_dic

compare = []
for key, val in new_dic.items():
    if np.all(np.array(val) > thre):
        compare.append((sum(val), *val, key))
compare = sorted(compare, reverse=True)

units = []
top = min([top, len(compare)])
for i in range(top):
    unit = compare[i][-1] # highly invariant unit
    print(f"Unit is: {unit}, the r-squared values are: {compare[i][1:-1]}")
    units.append(unit)

nav.npsave(units, nav.datapath, "gbp_AN", f"highFOIunits_hkey={hk}_thre={thre}.npy")