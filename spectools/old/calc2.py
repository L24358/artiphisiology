import numpy as np
import torch.nn as nn
import spectools.models.models as mdl

class Empty_layer():
    def __init__(self) -> None:
        self.kernel_size = 1
        self.stride = 1

def get_RF(rf, s, k):
    """
    @ Args:
        - rf (int): receptive field of previous layer
        - s (int): stride of previous layer
        - k (int): kernel size of this layer
    """
    def get_item(x):
        if isinstance(x, tuple): return x[0]
        return x
    
    rf, s, k = get_item(rf), get_item(s), get_item(k)
    return rf*k - (rf-s)*(k-1)

def get_RF_wrapper(model): # Conv2d and MaxPool only
    dic = {}
    prev_layer = Empty_layer()
    rf = prev_layer.kernel_size

    for i, layer in enumerate(model.features):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
            s = prev_layer.stride
            k = layer.kernel_size
            rf = get_RF(rf, s, k)
            prev_layer = layer
            dic[i] = rf
    return dic

def get_RF_resnet():
    model = mdl.get_resnet18()
    dic = {}
    prev_layer = Empty_layer()
    rf = prev_layer.kernel_size

    idx = [0, 3]
    pre_layers = [model.conv1, model.maxpool]
    for i, layer in enumerate(pre_layers):
        s = prev_layer.stride
        k = layer.kernel_size
        rf = get_RF(rf, s, k)
        prev_layer = layer
        dic[idx[i]] = rf
    i = idx[-1]

    main_layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for layer in main_layers:
        for bb in range(2): # per basic-block
            i += 1
            for child in [layer[bb].conv1, layer[bb].conv2]:
                s = prev_layer.stride
                k = child.kernel_size
                rf = get_RF(rf, s, k)
                prev_layer = child
            dic[i] = rf
    return dic

if __name__ == "__main__":
    model = mdl.get_vgg16()
    dic = get_RF_wrapper(model)
    print(dic)