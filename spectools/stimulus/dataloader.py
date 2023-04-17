import os
import torch
import numpy as np
import pandas as pd
import handytools.navigator as nav
import torchvision.transforms as trans
from torchvision.io import read_image
from torch.utils.data import Dataset

class Imagenette(Dataset):
    def __init__(self, transform = None):
        self.img_paths = nav.pklload("/src", "data", "imagenette", "train_images.pkl")
        self.img_labels = nav.pklload("/src", "data", "imagenette", "train_labels.pkl")
        self.folderpath = os.path.join("/src", "data", "imagenette", "train")
        self.transform = self.init_transform()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(os.path.join(self.folderpath, self.img_paths[idx]))
        image = self.transform(image)
        label = self.img_labels[idx]
        return image.type(torch.float), label
    
    def init_transform(self):
        return trans.Compose([trans.CenterCrop(227)])


if __name__ == "__main__":
    if False:
        img_path = os.path.join("/src", "data", "imagenette", "train")

        paths = []
        labels = []
        for label, dir in enumerate(sorted(os.listdir(img_path))):
            for file in os.listdir(os.path.join(img_path, dir)):
                paths.append(os.path.join(dir, file))
                labels.append(label)
        
        nav.pklsave(paths, "/src", "data", "imagenette", "train_images.pkl")
        nav.pklsave(labels, "/src", "data", "imagenette", "train_labels.pkl")