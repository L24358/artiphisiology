import numpy as np
import handytools.navigator as nav

class IQue():
    def __init__(self):
        self.master_dic = {}

    def __call__(self, datatype, mtype, key, unit, preprocess=2):
        if (datatype, mtype, key, preprocess) not in self.master_dic.keys(): self.load(datatype, mtype, key, preprocess)
        metrics = self.master_dic[(datatype, mtype, key, preprocess)]
        r2_filtered = np.maximum(metrics[unit][2], 0)
        return r2_filtered

    def load(self, datatype, mtype, key, preprocess): # load and store to avoid re-loading
        metrics = nav.pklload("/src", "results", f"{datatype}_{mtype}", f"key={key}_preproc={preprocess}", "fit_metrics.pkl")
        self.master_dic[(datatype, mtype, key, preprocess)] = metrics