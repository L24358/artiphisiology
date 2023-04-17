import torch
import numpy as np
from spectools.models.models import get_vgg16
from spectools.models.deconvnet import VGG16_deconv

key = 11
unit = 435

model = get_vgg16(hidden_keys=[key])
X = torch.randint(low=0, high=100, size=(2, 3, 227, 227)) # (B, C, H, W)
model(X)
R = model.hidden_info[key][0][:, unit:unit+1, ...] # single channel
pool_indices = model.pool_indices
output_size = model.output_size

print("Start input into decovnet")
del model
dmodel = VGG16_deconv()
Y = dmodel(R, key, unit, pool_indices, output_size)
import pdb; pdb.set_trace()