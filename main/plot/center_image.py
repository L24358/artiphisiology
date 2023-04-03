import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import shift

tot_pxl = 1024

def get_centroid(image_array):
    coors = []
    for i in range(tot_pxl):
        for j in range(tot_pxl):
            vals = image_array[i][j]
            if 0 in vals: coors.append((i, j))
    centroid = np.mean(coors, axis=0)
    return centroid

def not_in(a, lists):
    for i in lists:
        if np.all(a == i): return False
    return True

true_center = tot_pxl/2.0
for s in [9]: # range(51)
    image = Image.open(f"/src/data/shapes_uncentered/idx={s}_pxl={tot_pxl}.png")
    image_array = np.asarray(image)
    
    temp = image_array.reshape(-1, 4)
    uniques = []
    for i in temp:
        if not_in(i, uniques): uniques.append(i)
    import pdb; pdb.set_trace()

    centroid = get_centroid(image_array)
    print(centroid)

    dx, dy = true_center - centroid
    new_image_array = shift(image_array, [dy, dx, 0], mode="nearest")
    new_centroid = get_centroid(new_image_array)
    print(new_centroid)

    new_image = Image.fromarray(new_image_array)
    new_image.save(f"/src/data/shapes_centered/idx={s}_pxl={tot_pxl}.png")
