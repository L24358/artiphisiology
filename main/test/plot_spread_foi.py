import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis

hkey = 3

def get_r2(datatype, mtype, key, preprocess):
    """Get filtered r-squared values (i.e. ReLU(r2))."""
    metrics = nav.pklload(nav.homepath, "results", f"{datatype}_{mtype}", f"key={key}_preproc={preprocess}", "fit_metrics.pkl")
    metrics_array = np.asarray(list(metrics.values())) # shape = (# responsive units, # metrics)
    r2_filtered = metrics_array[:,0]
    return r2_filtered, list(metrics.keys())

r2_foi, idx_foi = get_r2("FOI", "AN", hkey, 2)
spread = nav.npload(nav.datapath, "temp", "corr_spread.npy")
plt.plot(r2_foi, spread[idx_foi], "k.")
vis.savefig()