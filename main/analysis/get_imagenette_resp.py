"""
Store responses to Imagenette.
"""

import gc
import torch
import torch.nn as nn
import handytools.navigator as nav
from torch.utils.data import DataLoader
from torch.optim import Adam
from spectools.models.models import get_vgg16
from spectools.models.deconvnet import VGG16_deconv
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
key = 11
unit = 435
bs = 128
device = "cuda:0"
torch.manual_seed(42)

# define dataset, models
dataset = Imagenette()
train_dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
model = get_vgg16(hidden_keys=[key]).to(device)

start, end, flag_output = 0, 100, True
model.eval()
for i, data in enumerate(train_dataloader):

    print(i)
    if i >= start:
        # get input, reset optimizer
        image, label, img_idx = data
        model(image.to(device), premature_quit = True, filt = lambda x: x[:, unit:unit+1, ...]) # feed image into model

        nav.pklsave(model.hidden_info, nav.datapath, f"hresp_imagenette_seed={42}_bs={bs}", f"key={key}_unit={unit}_B={i}.pkl")
        nav.pklsave(model.pool_indices, nav.datapath, f"poolidx_imagenette_seed={42}_bs={bs}", f"key={key}_unit={unit}_B={i}.pkl")
        if flag_output: 
            nav.pklsave(model.output_size, nav.datapath, f"outsize_imagenette_seed={42}_bs={bs}", f"key={key}_unit={unit}.pkl")

        del image, label, img_idx
        model.reset_storage()
        gc.collect()
        flag_output = False

    if i > end: break



