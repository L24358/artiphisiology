import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
from spectools.metrics.metrics import cubic_spline, curvature, angle, double_exp_wrap, responsive
from spectools.solvers import GNSolver

# hyperparameters
c_num =  0

shape_coor = nav.pklload("/src", "data", "stimulus", "shape_coor.pkl")
Rs = nav.npload("/src", "data", "responses", "CR_stim=shape_key=8.npy") # (C, S)
_, idxs = responsive(Rs, 50)

for idx in idxs:
    R = Rs[idx]

    xbundle = []
    for s in range(len(shape_coor)):
        anchors = np.array(shape_coor[s]).reshape(-1, 2)
        ps, dps, ddps = cubic_spline(anchors)

        ks, ts = [], []
        for i in range(len(ps)):
            k = curvature(dps[i], ddps[i])
            t = angle(ps[i])
            ks.append(k)
            ts.append(t)
        ks_ts = np.vstack((ks, ts)).T
        xbundle.append(ks_ts)

    gns = GNSolver(fit_function = double_exp_wrap)
    coefs = gns.fit(xbundle, R, init_guess=[-0.75, 0.5, np.pi, np.pi/2, 1])
    R_pred = gns.predict(xbundle)

    ax = vis.simpleaxis(None)
    ax.plot(R, R_pred, "k.")
    plt.xlabel("R (real)"); plt.ylabel("R (prediction)")
    vis.savefig(f"apc_idx={idx}.png")
    nav.npsave(coefs, "/src", "data", "parameters", f"apccoefs_idx={idx}.npy")