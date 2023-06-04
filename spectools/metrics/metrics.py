import numpy as np

def responsive(r, criterion="nonzero_by_thre", **kwargs):
    kw = {"thre": 20, "perc": 0.1, "abs": False}
    kw.update(kwargs)

    if kw["abs"]: r = abs(r)
    if criterion == "nonzero_by_thre":
        nonzeros = np.count_nonzero(np.array(r) >= 1e-3)
        if nonzeros >= kw["thre"]: return True # if non-zero responses are larger than threshold
        return False
    
    elif criterion == "nonzero_by_perc":
        zeros = list(r).count(0)
        if zeros <= len(r)*kw["perc"]: return True # if zero responses are smaller than percentage
        return False
    
def get_prs(R_fills, R_outlines, hkeys, verbose=False):
    prs = {}
    respdic = {}
    for hkey in hkeys:
        R_fill = R_fills[hkey]
        R_outline = R_outlines[hkey]

        prs[hkey] = np.array([])
        respdic[hkey] = []
        
        for unit in range(R_fill.shape[0]):
            resp = responsive(R_fill[unit]) and responsive(R_outline[unit])
            if resp:
                pr = np.corrcoef(R_fill[unit], R_outline[unit])[0][1]
                prs[hkey] = np.append(prs[hkey], pr)
                respdic[hkey].append(int(unit))
            else:
                prs[hkey] = np.append(prs[hkey], np.nan)
                if verbose: print(f"Layer {hkey}, unit {unit} not responsive!")
    return prs, respdic