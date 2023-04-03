import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from spectools.metrics.metrics import cubic_spline

def get_image(array): return Image.fromarray(array)

def get_shape(invec, tot_pxl):
    anchors = np.array(invec).reshape(-1, 2)
    ps, dps, ddps = cubic_spline(anchors)

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(tot_pxl*px, tot_pxl*px), facecolor="k")
    plt.axis("off")
    plt.fill(*ps.T, 'w')
    plt.xlim(-4, 4); plt.ylim(-4, 4)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_array, get_image(image_array)
