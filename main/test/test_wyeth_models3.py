import torch
import numpy as np
import matplotlib.pyplot as plt
import spectools.models.models as mdl
import handytools.navigator as nav
import handytools.visualizer as vis
import spectools.basics as bcs
from torchvision import models
from torchvision.models import AlexNet_Weights

exec(open('/home2/belleliu/artiphisiology/data/wyeth_foi/d05_shape_util.py').read())

device = "cpu"

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

hkey = 2
model = mdl.get_vgg16b(hidden_keys=[hkey])
model.to(device)
model(fill)
model(hollow)
R_fill = model.hidden_info[hkey][0].cpu() # shape = (724,192,27,27)
Rc_fill = bcs.get_center_response(R_fill) # shape = (192, 724)
R_hollow = model.hidden_info[hkey][1].cpu() # shape = (724,192,27,27)
Rc_hollow = bcs.get_center_response(R_hollow) # shape = (192, 724)

all_pr = []
for unit in range(64):
  x = Rc_fill[unit]
  y = Rc_hollow[unit]
  pr = np.corrcoef(x, y)[0][1]
  print(unit, pr)
  all_pr.append(pr)

print(np.mean(all_pr))
import pdb; pdb.set_trace()
# temp = rc[:, :, 13, 13]
# temp2 = Rc.swapaxes(0, 1).flatten()
# plt.plot(temp.cpu().detach().numpy().flatten(), temp2, "k.")
# vis.savefig()