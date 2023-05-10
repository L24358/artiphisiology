import torch
import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis
import spectools.basics as bcs
import spectools.models.models as mdl

def vstack_alt(lst1, lst2): # alternatively stacking elements from lst1 and 2
    return np.array([sub[item] for item in range(len(lst2)) for sub in [lst1, lst2]])

s_hollow = nav.npload(nav.datapath, "wyeth_foi", "hollow.npy")
s_fill = nav.npload(nav.datapath, "wyeth_foi", "fill.npy")
s_all = vstack_alt(s_fill, s_hollow) # shape = (724,227,227), fill goes first
s_all = np.expand_dims(s_all, 1)
s_all = np.tile(s_all, (1,3,1,1)) # shape = (724,3,227,227)

model = mdl.get_alexnet(hidden_keys=[2,3], in_place=False)
model(torch.from_numpy(s_all))
R = model.hidden_info[3][0] # shape = (724,192,27,27)
Rc = bcs.get_center_response(R) # shape = (192, 724)
Rw = nav.npload(nav.datapath, "wyeth_foi", "conv2_sr.npy")

plt.plot(Rc.flatten(), Rw.flatten(), "k.")
plt.xlabel("Conv2 resp, my network")
plt.ylabel("Conv2 resp, Wyeth's network")
vis.savefig()

import pdb; pdb.set_trace()
