import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
from spectools.metrics.metrics import IGV

hidden_key = 8
light = True
linewidth = 1
data_type = "rotated"
mtype = "VGG16"

folders = [f"{mtype}_dyrange_{data_type}_key={hidden_key}_lw={linewidth}"]

Rs = []
for scale in [0.25, 0.5, 1, 2, 4]:
    R = nav.npload("/src", "results", f"responses_{data_type}_hollow=0_lw={linewidth}_light={int(light)}_scale={scale}", f"{mtype}_CR_stim=shape_key={hidden_key}.npy")
    Rs.append(np.expand_dims(R, 0)) # (scale, units, images)
Rs = np.vstack(Rs)
Rs = np.swapaxes(Rs, 0, 1) # (units, scale, images)

for s in range(len(Rs)):
    x = Rs[s, 2]
    y1, y2, y3, y4 = Rs[s, 0], Rs[s, 1], Rs[s, 3], Rs[s, 4]
    plt.scatter(x, y1, color="r", label="scale=0.25")
    plt.scatter(x, y2, color="g", label="scale=0.5")
    plt.scatter(x, y3, color="b", label="scale=2")
    plt.scatter(x, y4, color="m", label="scale=4")
    minn = min([min(y1), min(y2), min(y3)])
    maxx = max([max(y1), max(y2), max(y3)])
    plt.plot([minn, maxx], [minn, maxx], "k--") # plot x=y
    plt.xlabel("Resp. to original stimuli"); plt.ylabel("Resp. to scaled stimuli"); plt.title(f"IGV: {IGV(Rs[s])}"); plt.legend()
    vis.savefig(f"idx={s}.png", folders=folders)


