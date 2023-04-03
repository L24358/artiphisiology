import numpy as np
import matplotlib.pyplot as plt
import spectools.basics as bcs
import spectools.visualization as vis
import handytools.navigator as nav
from scipy.ndimage import shift

tot_pxl = 227
true_center = tot_pxl/2.0
shape_coor = nav.pklload("/src", "data", "stimulus", "shape_coor.pkl")

for s in range(len(shape_coor)):
    image_array = vis.get_shape(shape_coor[s], tot_pxl)[0]
    centroid = bcs.get_centroid(image_array, tot_pxl)
    dx, dy = true_center - centroid

    new_image_array = shift(image_array, [dx, dy, 0], mode="nearest")
    new_image = vis.get_image(new_image_array)
    new_image.save(f"/src/data/shapes_centered/idx={s}_pxl={tot_pxl}.png")
    nav.npsave(new_image_array, "/src", "data", "stimulus", f"idx={s}_pxl={tot_pxl}.npy")

    plt.close("all")
    

