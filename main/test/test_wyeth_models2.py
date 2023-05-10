import torch
import numpy as np
import matplotlib.pyplot as plt
import spectools.models.models as mdl
import handytools.navigator as nav
import handytools.visualizer as vis
import spectools.basics as bcs
from torchvision import models
from torchvision.models import AlexNet_Weights

def get_n01_alexnet(layname):
  mod = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
  
  if   (layname == 'conv1'):  i =  0
  elif (layname == 'conv2'):  i =  3
  elif (layname == 'conv3'):  i =  6
  elif (layname == 'conv4'):  i =  8
  elif (layname == 'conv5'):  i = 10
  else:
    print("*** GET_N01_ALEXNET:  Unknown layer " + layname)
    return None, 0
  
  m = mod.features[:i+1]
  zn = m[i].out_channels
  return m, zn

device = "cuda:0"
m, zn = get_n01_alexnet('conv2')
dstim = torch.rand(1, 3, 227, 227)

m = m.to(device)
ttstim = dstim.to(device).float()
rc = m.forward(ttstim) 

model = mdl.get_alexnet(hidden_keys=[2,3], in_place=False)
model.to(device)
model(ttstim)
R = model.hidden_info[3][0].cpu() # shape = (724,192,27,27)
Rc = bcs.get_center_response(R) # shape = (192, 724)

temp = rc[0, :, 13, 13]
temp2 = Rc.flatten()
plt.plot(temp.cpu().detach().numpy(), temp2, "k.")
vis.savefig()