"""
Finding (Questionable) Receptive Fields by averaging over guided backprop image batches.

@ TODO:
    - Pruning
"""

import numpy as np
import handytools.navigator as nav
import spectools.models.calc as calc
import spectools.models.models as mdl
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
hkey = 3

# load
dataset = Imagenette("train")
model = mdl.get_alexnet()
boundaries = calc.get_boundaries_wrap(model)[hkey] # This is wrong

# main
for unit in range(192):

    QRF = []
    for i in range(5): # len(dataset)
        R = nav.npload(nav.datapath, "results", "gbp_AN", f"R_hkey={hkey}_unit={unit}_idx={i}.npy") # shape = (h,w)
        ggrads = nav.npload(nav.datapath, "results", "gbp_AN", f"ggrad_hkey={hkey}_unit={unit}_idx={i}.npy") #shape = (227,227)

        R_flatten = R.flatten()
        for j in range(len(R_flatten)):
            id1, id2 = np.unravel_index(j, R.shape)
            leftx, rightx = boundaries[id1]
            lefty, righty = boundaries[id2]
            patch = ggrads[:, int(leftx):int(rightx), int(lefty):int(righty)] # the image patch that this location truly sees
            QRF.append(patch*R_flatten[j]) # (3, h, w)
    
    QRF = np.asarray(QRF).mean(axis=0) # TODO: Shape mismatch error because of boundary issues
    