import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import handytools.visualizer as vis
from spectools.metrics.metrics import cubic_spline

def get_image(array): return Image.fromarray(array)

def get_ps(invec):
    anchors = np.array(invec).reshape(-1, 2)
    ps, dps, ddps = cubic_spline(anchors)
    return ps

def get_shape(invec, tot_pxl, facecolor="w", edgecolor="w", linewidth=1, exist_ps=False):
    if exist_ps: ps = invec
    else: ps = get_ps(invec)

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(tot_pxl*px, tot_pxl*px), facecolor="k")
    plt.axis("off")
    plt.fill(*ps.T, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    plt.xlim(-4, 4); plt.ylim(-4, 4)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_array, get_image(image_array)

def show_img_and_grad(image, ggrads, figname, title=""):
    plt.figure()
    plt.subplot(1,2,1)
    img = image - image.min()   # Normalize the gradient image
    img = img /img.max()
    img = np.moveaxis(img.numpy(),0,-1)
    plt.imshow(img)  # Plot the original image

    gim = ggrads - ggrads.min()   # Normalize the gradient image
    gim = gim /gim.max()
    gim = np.moveaxis(gim,0,-1)  # Move 1st array dimension to end
    plt.subplot(1,2,2);
    plt.imshow(gim)
    plt.title(title)
    vis.savefig(figname)