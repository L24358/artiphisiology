import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
from sklearn.metrics import r2_score
from spectools.old.metrics import cubic_spline, curvature, angle, apc_obj_func, response_sparsity, double_exp_wrap
from spectools.old.solvers import DESolver

# hyperparameters
hidden_key = 10
c_num =  0

shape_coor = nav.pklload("/src", "data", "stimulus", "shape_coor.pkl")
Rs = nav.npload("/src", "data", "responses", f"CR_stim=shape_key={hidden_key}.npy") # (C, S)
_, idxs = response_sparsity(Rs) # or responsive, 50

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

    bounds = [(-0.5, 1), (0.01, 0.98), (-np.pi, np.pi), (0, np.pi), (-1e5, 1e5)]
    des = DESolver(apc_obj_func, bounds, double_exp_wrap)
    coefs = des.fit(xbundle, R)
    R_pred = des.predict(xbundle)
    r2 = r2_score(R, R_pred)
    print("Coefs: ", coefs)

    ax = vis.simpleaxis(None)
    ax.plot(R, R_pred, "k.")
    plt.xlabel("R (real)"); plt.ylabel("R (prediction)"); plt.title(f"r squared: {r2}")
    vis.savefig(f"apc2_idx={idx}.png")
    nav.npsave(coefs, "/src", "data", "parameters", f"apccoefs2_idx={idx}_key={hidden_key}.npy")