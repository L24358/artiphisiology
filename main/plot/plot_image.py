import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
from PIL import Image
from scipy.ndimage import shift
from spectools.metrics.metrics import cubic_spline, curvature, angle

tot_pxl = 1024
true_center = tot_pxl/2.0
shape_coor = nav.pklload("/src", "data", "stimulus", "shape_coor.pkl")

def get_centroid(image_array):
    coors = []
    for i in range(tot_pxl):
        for j in range(tot_pxl):
            vals = image_array[i][j]
            if 255 in vals: coors.append((i, j))
    centroid = np.mean(coors, axis=0)
    return centroid

def not_in(a, lists):
    for i in lists:
        if np.all(a == i): return False
    return True

for s in range(len(shape_coor)):
    anchors = np.array(shape_coor[s]).reshape(-1, 2)
    ps, dps, ddps = cubic_spline(anchors)

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(tot_pxl*px, tot_pxl*px), facecolor="k")
    plt.axis("off")
    plt.fill(*ps.T, 'w')
    plt.xlim(-4, 4); plt.ylim(-4, 4)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    centroid = get_centroid(image_array)
    dx, dy = true_center - centroid
    new_image_array = shift(image_array, [dx, dy, 0], mode="nearest")

    new_image = Image.fromarray(new_image_array)
    new_image.save(f"/src/data/shapes_centered/idx={s}_pxl={tot_pxl}.png")

    plt.close("all")
    

