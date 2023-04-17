import torch
import numpy as np
from spectools.models.models import get_vgg16
from spectools.models.deconvnet import VGG16_deconv

key = 11
unit = 435

print("Start defining deconv net")
dmodel = VGG16_deconv()
print("Finished defining model")
import pdb; pdb.set_trace()

X = torch.randint(low=0, high=100, size=(2, 3, 227, 227)) # (B, C, H, W)
model = get_vgg16(hidden_keys=[key])
print("Start input into VGG16")
model(X)
R = model.hidden_info[key][0]
Ridx = model.hidden_info.get_more(key, "maxpool2d_idx")[0]
import pdb; pdb.set_trace()

print("Start input into decovnet")
del model
dmodel(R, key, unit, Ridx)