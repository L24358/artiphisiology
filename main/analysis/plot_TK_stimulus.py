import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
import spectools.visualization as vis2

for i in range(393):
    image_array = nav.npload(nav.homepath, "data", "stimulus_TK", f"idx={i}_pxl=227.npy")
    image = vis2.get_image(image_array)
    plt.imshow(image)
    vis.savefig(f"idx={i}_pxl=227.png", folders=[nav.graphpath, "stimulus_TK"])