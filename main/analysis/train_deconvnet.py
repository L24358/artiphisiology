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

# define model, loss func, optimizer
dataset = Imagenette()
train_dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
dmodel = VGG16_deconv()
optimizer = Adam(dmodel.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# training loop
dmodel.train()

losses = []
for epoch in range(2):
    # print
    print("Epoch: ", epoch)

    for i, data in enumerate(train_dataloader):

        # get single channel output from model
        image, _, _ = data
        hidden_info = nav.pklsave("/src", "data", f"hresp_imagenette_seed={42}_bs={bs}", f"key={key}_unit={unit}_B={i}.pkl")
        pool_indices = nav.pklsave("/src", "data", f"poolidx_imagenette_seed={42}_bs={bs}", f"key={key}_unit={unit}_B={i}.pkl")
        output_size = nav.pklsave("/src", "data", f"outsize_imagenette_seed={42}_bs={bs}", f"key={key}_unit={unit}_B={i}.pkl")
        R = hidden_info[key][0] # select single channel

        # pass input into deconv model
        optimizer.zero_grad()
        image_pred = dmodel(R, key, unit, pool_indices, output_size)

        # calculate loss and step
        loss = loss_fn(image, image_pred)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        
        # print info
        if True: #i%100 == 0:
            print("Loss: ", loss.item())

import pdb; pdb.set_trace()