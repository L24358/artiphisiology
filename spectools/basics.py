import numpy as np
from handytools.catcher import AlgorithmError

def get_centroid(image_array, tot_pxl):
    coors = []
    for i in range(tot_pxl):
        for j in range(tot_pxl):
            vals = image_array[i][j]
            if 255 in vals: coors.append((i, j))
    centroid = np.mean(coors, axis=0)
    return centroid

def get_center_response(responses):
    if len(responses.shape) == 4: 
        B, C, H, W = responses.shape
        return responses[:, :, H // 2, W //2].squeeze().detach().numpy().swapaxes(0, 1) # shape = (C, B)
    elif len(responses.shape) == 2:
        B, C = responses.shape
        return responses.detach().numpy().T # shape = (C, B)
    else:
        raise AlgorithmError(f"The shape of responses is incorrect. Should have either 2 or 4 dimensions.")
    
def preprocess1(image_array, light, scale):
    """Conforms with the preprocessing steps documented in torch hub."""
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    image_array = image_array/255. # so that it is within the range [0, 1]
    out_array = (image_array - means)/stds

    if not light:
        out_array = (1+scale)*means - scale*out_array # reflection w.r.t. mean, then scale
    else:
        out_array = (1-scale)*means + scale*out_array
    return out_array

def preprocess2(image_array, light, scale):
    image_array = image_array/255. # dark BG (0), light image (1)
    if not light: image_array *= -1 # light BG (0), dark image (-1)
    image_array *= scale
    return image_array

def infer_nunits(n):
    l = [2**i for i in range(12)]
    d = abs(n-np.array(l))
    idx = list(d).index(min(d))
    if 2**idx > n: return 2**idx
    else: return 2**(idx+1)
