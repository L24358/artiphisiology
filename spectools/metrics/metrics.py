import numpy as np

def responsive(r, criterion="nonzero_by_thre", **kwargs):
    kw = {"thre": 20, "perc": 0.1}
    kw.update(kwargs)

    if criterion == "nonzero_by_thre":
        nonzeros = np.count_nonzero(np.array(r) >= 1e-3)
        if nonzeros >= kw["thre"]: return True # if non-zero responses are larger than threshold
        return False
    
    elif criterion == "nonzero_by_perc":
        zeros = list(r).count(0)
        if zeros <= len(r)*kw["perc"]: return True # if zero responses are smaller than percentage
        return False