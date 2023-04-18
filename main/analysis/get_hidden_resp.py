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
torch.manual_seed(42)

# define dataset, models
dataset = Imagenette()
train_dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
model = get_vgg16(hidden_keys=[key])

start, end = 0, 100
model.eval()
for i, data in enumerate(train_dataloader):

    if i >= start:
        # get input, reset optimizer
        image, label, img_idx = data
        model(image, premature_quit = True, filt = lambda x: x[:, unit:unit+1, ...]) # feed image into model

        nav.pklsave(model.hidden_info, "/src", "data", f"hresp_imagenette_seed={42}_bs={bs}", f"key={key}_unit={unit}_B={i}.pkl")
        nav.pklsave(model.pool_indices, "/src", "data", f"poolidx_imagenette_seed={42}_bs={bs}", f"key={key}_unit={unit}_B={i}.pkl")
        nav.pklsave(model.output_size, "/src", "data", f"outsize_imagenette_seed={42}_bs={bs}", f"key={key}_unit={unit}_B={i}.pkl")

        del image, label, img_idx
        model.reset_storage()
        gc.collect()

    if i > end: break



