"""
Preprocess images from imagenette.
"""

import os
import numpy as np
import handytools.navigator as nav
from spectools.stimulus.dataloader import Imagenette

if False: # step 1: store all the paths
    img_path = os.path.join(nav.datapath, "imagenette", "train")

    paths = []
    labels = []
    folders = nav.list_dir(img_path, os.path.isdir)
    for label, dir in enumerate(sorted(folders)):
        for file in os.listdir(os.path.join(img_path, dir)):
            paths.append(os.path.join(dir, file))
            labels.append(label)
    
    nav.pklsave(paths, nav.datapath, "imagenette", "train_images.pkl")
    nav.pklsave(labels, nav.datapath, "imagenette", "train_labels.pkl")

if True: # step 2: filter out single-channel images, need to also adjust the loading name in dataloaders.Imagenette()
    dataset = Imagenette()
    img_paths = dataset.img_paths
    img_labels = dataset.img_labels

    new_paths = []
    new_labels = []
    for i in range(len(dataset)):
        image, label, img_idx = dataset[i]
        if tuple(image.shape) == (3, 227, 227):
            new_paths.append(img_paths[i])
            new_labels.append(img_labels[i])
        del image, label, img_idx

        if i%1000 == 0: print("Progress: ", i)

    nav.pklsave(new_paths, nav.datapath, "imagenette", "train_images_filtered.pkl")
    nav.pklsave(new_labels, nav.datapath, "imagenette", "train_labels_filtered.pkl")

