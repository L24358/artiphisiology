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

fill = nav.npload(nav.datapath, "wyeth_foi", "fill.npy")
hollow = nav.npload(nav.datapath, "wyeth_foi", "hollow.npy")
fill = np.expand_dims(fill, axis=1)
hollow = np.expand_dims(hollow, axis=1)
fill = np.tile(fill, (1,3,1,1))
hollow = np.tile(hollow, (1,3,1,1))
fill = torch.from_numpy(fill).to(device).float()
hollow = torch.from_numpy(hollow).to(device).float()

# m, zn = get_n01_alexnet('conv2')
# dstim = torch.from_numpy(fill)

# m = m.to(device)
# ttstim = dstim.to(device).float()
# rc = m.forward(ttstim) 

model = mdl.get_alexnet(hidden_keys=[10], in_place=False)
model.to(device)
model(fill)
model(hollow)
R_fill = model.hidden_info[10][0].cpu() # shape = (724,192,27,27)
Rc_fill = bcs.get_center_response(R_fill) # shape = (192, 724)
R_hollow = model.hidden_info[10][1].cpu() # shape = (724,192,27,27)
Rc_hollow = bcs.get_center_response(R_hollow) # shape = (192, 724)

all_pr = []
for unit in range(256):
  x = Rc_fill[unit]
  y = Rc_hollow[unit]
  pr = np.corrcoef(x, y)[0][1]
  print(unit, pr)
  all_pr.append(pr)


import pdb; pdb.set_trace()
# temp = rc[:, :, 13, 13]
# temp2 = Rc.swapaxes(0, 1).flatten()
# plt.plot(temp.cpu().detach().numpy().flatten(), temp2, "k.")
# vis.savefig()