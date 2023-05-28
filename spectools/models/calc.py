import numpy as np
import torch.nn as nn
import spectools.models.models as mdl

def get_RF(model):
    if not isinstance(model, list): # if a normal model is passed in
        layers = model.features 
        adj = 0
    else: # if a list is passed in instead
        layers = model 
        adj = 1

    rfs = {}
    strides, kernels, Ls = [1], [], [] # stride at layer -1 is 1
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
            s = layer.stride
            k = layer.kernel_size
            if isinstance(s, tuple): s=s[0]
            if isinstance(k, tuple): k=k[0]
            strides.append(s)
            kernels.append(k)
            Ls.append(i)

    for ll in range(1, len(Ls)+adj):
        summ = 1
        for l in range(0, ll):
            summ += (kernels[l]-1)*np.prod(strides[:l+1])
        rfs[Ls[ll-1]] = summ
    return rfs

def get_RF_resnet(map_to=True):
    model = mdl.get_resnet18()
    layers_of_interest = []

    pre_layers = [model.conv1, model.maxpool]
    for i, layer in enumerate(pre_layers): layers_of_interest.append(layer)

    main_layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for layer in main_layers:
        for bb in range(2): # per basic-block
            for child in [layer[bb].conv1, layer[bb].conv2]: layers_of_interest.append(child)

    rfs = get_RF(layers_of_interest)
    if not map_to: return rfs
    else:
        mapdic = {0:0, 3:1, 4:3, 5:5, 6:7, 7:9, 8:11, 9:13, 10:15, 11:17}
        newrfs = {}
        for key in mapdic.keys():
            newrfs[key] = rfs[mapdic[key]]
        return newrfs
    

if __name__ == "__main__":
    dic = get_RF_resnet()
    import pdb; pdb.set_trace()