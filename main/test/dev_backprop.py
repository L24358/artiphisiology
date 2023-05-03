import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import handytools.visualizer as vis
import spectools.models.models as mdl
from torch.autograd import Variable
from torch.utils.data import DataLoader
from spectools.models.deconvnet import GuidedBackprop
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
hkey = 8 # layer of interest
bs = 1
unit = 255

# load
mod = mdl.get_alexnet()
GBP = GuidedBackprop(mod)
dataset = Imagenette("train")
train_dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

for i, data in enumerate(train_dataloader):
    print("Batch ", i)
    image, _, _ = data
    tt_var = Variable(image, requires_grad=True)
    ggrads = GBP.generate_gradients(tt_var, hkey, unit)
    gim = ggrads - ggrads.min()   # Normalize the gradient image
    gim = gim /gim.max()
    gim = np.moveaxis(gim,0,-1)  # Move 1st array dimension to end
    plt.subplot(1,2,2);
    plt.imshow(gim)
    vis.savefig()
    import pdb; pdb.set_trace()