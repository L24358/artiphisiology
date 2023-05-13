import numpy as np
import handytools.navigator as nav
import matplotlib.pyplot as plt
import handytools.visualizer as vis

mtype = "AN"
hidden_key = 3
small = False
start = 255

R = nav.npload(nav.homepath, "results", f"responses_{mtype}", f"TKtexture_hkey={hidden_key}.npy")
R2 = nav.npload(nav.homepath, "results", f"responses_{mtype}", f"TKtexturecolor_hkey={hidden_key}.npy")
cs = nav.npload(nav.datapath, "temp", "corr_spread.npy")
foi = nav.npload(nav.datapath, "temp", f"highFOIunits_hkey={hidden_key}_thre={0.8}.npy")

R_std = []
in_foi = []
for unit in range(len(R)):
    temp = []
    for i in range(225-start, 393-start, 8):
        if not small: idx = list(range(i, i+4))
        else: idx = list(range(i+4, i+8))

        resp = R[unit][idx]
        temp.append(np.std(resp))
    R_std.append(np.mean(temp))

    if unit in foi: in_foi.append(1)
    else: in_foi.append(0)

R_std2 = []
for unit in range(len(R)): # 28
    temp = []
    for i in range(225-start, (393-start)*6, 6):
        idx = list(range(i, i+6))

        resp = R2[unit][idx]
        if np.mean(resp) > 0:
            temp.append(np.std(resp))
    if temp != []: R_std2.append(np.mean(temp))
    else: R_std2.append(0)


# plt.scatter(-np.array(R_std), R_std2, c=in_foi)
plt.scatter(cs, R_std2)
vis.savefig()

import pdb; pdb.set_trace()