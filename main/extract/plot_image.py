import numpy as np
import matplotlib.pyplot as plt
import spectools.basics as bcs
import spectools.visualization as vis
import handytools.navigator as nav
from scipy.ndimage import shift

# hyperparameters
hollow = True
tot_pxl = 227
linewidth = 1
first_run = False # first_run should only be used if hollow=False

# load data
true_center = tot_pxl/2.0
shape_coor = nav.pklload("/src", "data", "stimulus", "shape_coor.pkl")
filepath = ["/src", "data", f"stimulus_centered_hollow={int(hollow)}_lw={linewidth}"]
nav.mkfile(*filepath)
nav.mkfile(f"/src/graphs/shapes_centered_hollow={int(hollow)}_lw={linewidth}/")
if not first_run:
    shift_dic = nav.pklload("/src", "data", "stimulus", f"shift_filled_pxl={tot_pxl}_lw={linewidth}.pkl") # key: idx, value: shift [dx, dy, 0] for each idx
else:
    shift_dic = {}

# hollow or full
if hollow:
    facecolor, edgecolor = "k", "w"
else:
    facecolor, edgecolor = "w", "w"

# generate shape and center data
for s in range(len(shape_coor)):
    image_array = vis.get_shape(shape_coor[s], tot_pxl, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)[0]
    centroid = bcs.get_centroid(image_array, tot_pxl)
    dx, dy = true_center - centroid
    if first_run: shift_dic[s] = [dx, dy, 0]

    new_image_array = shift(image_array, shift_dic[s], mode="nearest")
    new_image = vis.get_image(new_image_array)
    new_image.save(f"/src/graphs/shapes_centered_hollow={int(hollow)}_lw={linewidth}/idx={s}_pxl={tot_pxl}.png")
    nav.npsave(new_image_array, *filepath, f"idx={s}_pxl={tot_pxl}.npy")

    plt.close("all")

if first_run: nav.pklsave("/src", "data", "stimulus", f"shift_filled_pxl={tot_pxl}_lw={linewidth}.pkl")