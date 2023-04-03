import numpy as np

def get_centroid(image_array, tot_pxl):
    coors = []
    for i in range(tot_pxl):
        for j in range(tot_pxl):
            vals = image_array[i][j]
            if 255 in vals: coors.append((i, j))
    centroid = np.mean(coors, axis=0)
    return centroid

def get_center_response(responses):
    B, C, H, W = responses.shape
    return responses[:, :, H // 2, W //2].squeeze().detach().numpy().swapaxes(0, 1) # shape = (C, B)