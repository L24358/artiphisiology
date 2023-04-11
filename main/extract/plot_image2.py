"""
Add rotation to existing images.
"""
import numpy as np
import matplotlib.pyplot as plt
import spectools.basics as bcs
import spectools.visualization as vis
import handytools.manipulator as mnp
import handytools.navigator as nav
from scipy.ndimage import shift

# hyperparameters
hollow = True
tot_pxl = 227
linewidth = 1

# load data
true_center = tot_pxl/2.0
rot_info = nav.pklload("/src", "data", "stimulus", "shape_info.pkl")["rotation"]
shape_coor = nav.pklload("/src", "data", "stimulus", "shape_coor.pkl")
filepath = ["/src", "data", f"stimulus_rotated_hollow={int(hollow)}_lw={linewidth}"]
nav.mkfile(f"/src/graphs/shapes_rotated_hollow={int(hollow)}_lw={linewidth}/")

# hollow or full
if hollow:
    facecolor, edgecolor = "k", "w"
else:
    facecolor, edgecolor = "w", "w"

# rotate and plot image
for s in range(51):
    ps = vis.get_ps(shape_coor[s])

    for r in range(rot_info[s]):
        theta = 2*np.pi/8*r
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        ps_rotated = np.matmul(R, ps.T).T

        new_image_array = vis.get_shape(ps_rotated, tot_pxl, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, exist_ps=True)[0]
        new_image = vis.get_image(new_image_array)
        new_image.save(f"/src/graphs/shapes_rotated_hollow={int(hollow)}_lw={linewidth}/idx={s}_pxl={tot_pxl}_r={r}.png")
        nav.npsave(new_image_array, *filepath, f"idx={s}_pxl={tot_pxl}_r={r}.npy")

        plt.close("all")