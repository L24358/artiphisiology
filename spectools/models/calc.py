import numpy as np
import torch.nn as nn
import spectools.models.models as mdl

def get_RF(model):
    strides, kernels, Ls = [1], [], [] # stride at layer -1 is 1
    for i, layer in enumerate(model.features):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
            s = layer.stride
            k = layer.kernel_size
            if isinstance(s, tuple): s=s[0]
            if isinstance(k, tuple): k=k[0]
            strides.append(s)
            kernels.append(k)
            Ls.append(i)
            
    rfs = {}
    for ll in range(1, len(Ls)):
        summ = 1
        for l in range(0, ll):
            summ += (kernels[l]-1)*np.prod(strides[:l+1])
        rfs[Ls[ll-1]] = summ
    return rfs

if __name__ == "__main__":
    model = mdl.get_alexnet()
    get_RF(model)