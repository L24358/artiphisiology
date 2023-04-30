import os
import torch
import numpy as np
import pandas as pd
import handytools.navigator as nav
import torchvision.transforms as trans
from torchvision.io import read_image
from torch.utils.data import Dataset

class Imagenette(Dataset):
    def __init__(self, tpe, transform = None):
        self.img_paths = nav.pklload(nav.datapath, "imagenette", f"{tpe}_images_filtered.pkl")
        self.img_labels = nav.pklload(nav.datapath, "imagenette", f"{tpe}_labels_filtered.pkl")
        self.folderpath = os.path.join(nav.datapath, "imagenette", tpe)
        self.transform = self.init_transform(transform)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(os.path.join(self.folderpath, self.img_paths[idx]))
        image = self.transform(image.float())
        label = self.img_labels[idx]
        return image.type(torch.float), label, idx
    
    def init_transform(self, transform):
        if transform == None: 
            return trans.Compose([trans.CenterCrop(227)])
        else:
            return transform
