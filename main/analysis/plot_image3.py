"""
Scale basic + rotated image shapes.
"""

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import spectools.basics as bcs
import spectools.visualization as vis
import handytools.manipulator as mnp
import handytools.navigator as nav
from scipy.ndimage import shift

# hyperparameters
hollow = False
tot_pxl = 227
linewidth = 1
d = 90

# load data
true_center = tot_pxl/2.0
rot_info = nav.pklload(nav.datapath, "stimulus", "shape_info.pkl")["rotation"]
rot_idx = nav.pklload(nav.datapath, "stimulus", "shape_index.pkl") # max:361
shape_coor = nav.pklload(nav.datapath, "stimulus", "shape_coor.pkl")
filepath = [nav.datapath, f"stimulus_rotated_hollow={int(hollow)}_lw={linewidth}"]
# nav.mkfile(nav.graphpath + f"shapes_scaled_hollow={int(hollow)}_lw={linewidth}/")

# hollow or full
if hollow:
    facecolor, edgecolor = "k", "w"
else:
    facecolor, edgecolor = "w", "w"

# get scale factor
image_array = nav.npload(*filepath, f"idx={1}_pxl={tot_pxl}_r={0}.npy")
diameter = max(image_array[..., 0].sum(axis=1)/255.) # default: 71.753
resize_factor = int(227*d/diameter)
transform = T.Compose([T.Resize(resize_factor),
                        T.CenterCrop(227),
                        ])

# scale and plot image
for i in rot_idx.keys():
    s, r = rot_idx[i]
    image_array = nav.npload(*filepath, f"idx={s}_pxl={tot_pxl}_r={r}.npy")
    image = vis.get_image(image_array)
    resize_image = transform(image)
    resize_image_array = np.array(resize_image)
    nav.npsave(resize_image_array, nav.datapath, f"stimulus_scaled={d}_hollow={int(hollow)}_lw={linewidth}", f"idx={s}_pxl={tot_pxl}_r={r}.npy")