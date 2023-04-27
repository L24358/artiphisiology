import torch.nn as nn

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
    

if __name__ == "__main__":
    import spectools.models.models as mdl
    model = mdl.get_alexnet()
    get_RF_wrap(model)