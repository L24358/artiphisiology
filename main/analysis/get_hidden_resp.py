import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from spectools.models.models import get_vgg16
from spectools.models.deconvnet import VGG16_deconv
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
key = 11
unit = 435
torch.manual_seed(42)

# define dataset, models
dataset = Imagenette()
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
model = get_vgg16(hidden_keys=[key])

model.eval()
for i, data in enumerate(train_dataloader):
    # get input, reset optimizer
    image, label, img_idx = data
    model(image, premature_quit = True, filt = lambda x: x[:, unit:unit+1, ...]) # feed image into model
    R = model.hidden_info[key][0][:, unit:unit+1, ...] # select single channel


