"""
Works after trace_unit3.py
"""
import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
import spectools.models.models as mdl

# hyperparameters
key = 11
unit = 435
N = 256
B = 73

diffs = {}
for n in range(N):
    diffs[n] = []

    for i in range(B):
        diff = nav.npload(nav.datapath, "results", "subtraction_VGG16", f"imagenette_unit={unit}_key={key}_preunit={n}_B={i}.npy")
        diffs[n] += list(diff)

diff_array = []
for k in diffs.keys(): diff_array.append(diffs[k])
diff_array = np.array(diff_array)
diff_mean = diff_array.mean(axis=1)
diff_std = diff_array.std(axis=1)
idx = np.flip(np.argsort(diff_mean)) # index from largest to smallest unit
nav.npsave(idx, nav.datapath, "results", "subtraction_VGG16", "respidx.npy")

params = mdl.get_parameters("vgg16")
weights = params["features.11.weight"].detach().numpy()[unit].squeeze()
mags = []
for filt in weights: # shape = (256, 3, 3)
    mag = np.abs(filt).sum()
    mags.append(mag)
mags = np.array(mags)
nav.npsave(idx, nav.datapath, "results", "subtraction_VGG16", "filtidx.npy")

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131)
ax1.hist(diff_mean, color="k")
ax1.set_xlabel("mean response difference"); ax1.set_ylabel("count")
ax2 = fig.add_subplot(132)
ax2_twin = ax2.twinx()
ax2.plot(range(1, N+1), mags[idx], color="r", linestyle="-", marker=".")
ax2_twin.set_xlabel("rank"); ax2_twin.set_ylabel("response difference", color="k"); ax2.set_ylabel("filter magnitude", color="r")
ax2_twin.errorbar(range(1,N+1), diff_mean[idx], diff_std[idx], color="k", linestyle="-", marker=".", ecolor="b")
ax3 = fig.add_subplot(133)
ax3.scatter(mags, diff_mean, color="k")
ax3.set_xlabel("filter magnitude"); ax3.set_ylabel("mean response difference")
plt.suptitle(f"Response difference for layer {key}, unit {unit}")
vis.savefig()
