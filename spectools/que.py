import numpy as np
import handytools.navigator as nav

def Ique(datatype, mtype, key, unit, preprocess=2):
    metrics = nav.pklload("/src", "results", f"{datatype}_{mtype}", f"key={key}_preproc={preprocess}", "fit_metrics.pkl")
    metrics_array = np.asarray(list(metrics.values())) # shape = (# responsive units, # metrics)
    r2_filtered = np.maximum(metrics_array[:,2], 0) # ReLU(r^2)
    return r2_filtered[unit] # this needs to be adjusted