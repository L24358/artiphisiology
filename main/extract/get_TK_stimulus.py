import cv2
import handytools.visualizer as vis
import handytools.navigator as nav
from spectools.visualization import get_image

foldername = "/src/data/TK393"
constraint = lambda f: ("ShapeTexture" in f) and ("txt" in f)

count = 0
for f in nav.list_dir(foldername, constraint):
    data = nav.read_row(f, foldername, nav.empty)
    image_array = nav.to_float(data[2]).reshape(227, 227)
    nav.npsave(image_array, "/src", "data", "stimulus_TK", f"idx={count}_pxl=227.npy")
    count += 1

    cv2.imwrite(f"/src/graphs/stimulus_TK393/idx={count}_pxl=227.png", image_array/100*256)