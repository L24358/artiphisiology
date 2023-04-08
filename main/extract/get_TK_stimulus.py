import cv2
import handytools.visualizer as vis
import handytools.navigator as nav
from spectools.visualization import get_image
from collections import OrderedDict

foldername = "/src/data/TK393"
constraint = lambda f: ("ShapeTexture" in f) and ("txt" in f)

def sortfile(files):
    dic = {}
    for f in files:
        num = int(f.split(".")[0].split("_")[1])
        dic[num] = f
    dic = OrderedDict(sorted(dic.items()))
    return dic.values()

count = 0
files = sortfile(nav.list_dir(foldername, constraint))
for f in files:
    data = nav.read_row(f, foldername, nav.empty)
    image_array = nav.to_float(data[2]).reshape(227, 227)
    nav.npsave(image_array, "/src", "data", "stimulus_TK", f"idx={count}_pxl=227.npy")
    count += 1

    cv2.imwrite(f"/src/graphs/stimulus_TK393/idx={count}_pxl=227.png", image_array/100*256)