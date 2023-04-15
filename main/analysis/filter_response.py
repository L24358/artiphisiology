import numpy as np
import handytools.navigator as nav
from spectools.metrics.metrics import response_sparsity

def constraint(f):
    deter1 = "responses" in f
    deter2 = "_scale=1" in f
    return deter1 and deter2
response_folders = nav.list_dir("/src/data/", constraint)

Rs = []
for folder in response_folders:
    R = nav.npload("/src/data/", folder, "CR_stim=shape_key=8.npy")
    Rs.append(R)
Rs = np.hstack(Rs)
_, idx = response_sparsity(Rs)
nav.npsave(idx, "/src", "data", "responses", "sparse_idx.npy")