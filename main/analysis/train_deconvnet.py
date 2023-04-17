import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from spectools.models.models import get_vgg16
from spectools.models.deconvnet import VGG16_deconv
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
key = 11
unit = 435

# define dataset, models
dataset = Imagenette()
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
model = get_vgg16(hidden_keys=[key])
dmodel = VGG16_deconv()

# define loss function, optimizer
optimizer = Adam(dmodel.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# training loop
dmodel.train()

losses = []
for epoch in range(2):
    # print
    print("Epoch: ", epoch)

    for i, data in enumerate(train_dataloader):
        # get input, reset optimizer
        image, label = data
        optimizer.zero_grad()

        ## TODO: the results of this part can be saved!
        # get single channel output from model
        model(image, premature_quit = True) # feed image into model
        R = model.hidden_info[key][0][:, unit:unit+1, ...] # select single channel
        pool_indices = model.pool_indices
        output_size = model.output_size
        model.reset_storage()

        # pass input into deconv model
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