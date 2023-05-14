import numpy as np
import torch.nn as nn
from copy import deepcopy
from math import floor

def get_RF(k, s, m, idx=0):
    if k == None: return m[idx]
    if isinstance(k, tuple): k = k[idx]
    if isinstance(s, tuple): s = s[idx]
    if isinstance(m, tuple): m = m[idx]
    base = k*m
    overlap = (k-s)*(m-1)
    return base - overlap

def get_RF_wrap(model, verbose=False): # Conv2d and MaxPool only
    prev_rf, prev_layer, s, dic = None, None, None, {}
    for i, layer in enumerate(model.features):

        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
            s = prev_layer.stride if prev_layer != None else s
            m = layer.kernel_size
            rf = get_RF(prev_rf, s, m)
            prev_layer = layer
            prev_rf = rf

            if verbose: print(f"Layer {i} has RF={rf}.")

        dic[i] = rf
    return dic

def get_left(prevs, m, p, s2):
    assert s2 <= m # in this calculation, padding ``s2`` cannot exceed kernel size ``m``

    h = len(prevs)
    l = list(np.ones(s2)*prevs[0]) + list(prevs) + list(np.ones(s2)*prevs[-1])
    idxs = l[:len(l)-(m-1):s2]
    idxs = list(map(int, idxs))
    lefts = np.array(l)[idxs]
    return lefts

def get_right(prevs, m, p, s2):
    assert s2 <= m # in this calculation, padding ``s2`` cannot exceed kernel size ``m``

    h = len(prevs)
    l = list(np.ones(s2)*prevs[0]) + list(prevs) + list(np.ones(s2)*prevs[-1])
    idxs = l[m-1:len(l):s2]
    idxs = list(map(int, idxs))
    rights = np.array(l)[idxs]
    return rights

def get_boundaries_wrap(model, inp_size=227, verbose=False):
    # left is included, right is excluded, like standard python indexing
    prev_lefts = list(range(inp_size))
    prev_rights = np.array(prev_lefts).astype(np.int32) + 1

    dic = {} # key: layer, value: boundary pairs
    for i, layer in enumerate(model.features):

        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
            m, p, s2 = layer.kernel_size, layer.padding, layer.stride
            if isinstance(m, tuple): m = m[0]
            if isinstance(p, tuple): p = p[0]
            if isinstance(s2, tuple): s2 = s2[0]

            lefts = get_left(prev_lefts, m, p, s2)
            rights = get_right(prev_rights, m, p, s2)
            dic[i] = list(zip(lefts, rights))
    return dic

def get_images_from_loader(dataloader, idxs, size=(3,227,227)):
    res = np.empty((len(idxs),*size))
    idxs = list(idxs)
    idxs_copy = list(deepcopy(idxs))

    for i, data in enumerate(dataloader):
        for j in idxs:
            if i == j:
                placement = idxs_copy.index(j)
                res[placement] = data[0]
                idxs.remove(j)
    return res

    

if __name__ == "__main__":
    import spectools.models.models as mdl
    model = mdl.get_alexnet()
    get_RF_wrap(model, verbose=True)