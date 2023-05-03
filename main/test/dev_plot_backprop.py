import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import handytools.visualizer as vis
import spectools.models.models as mdl
from torch.autograd import Variable
from spectools.models.deconvnet import GuidedBackprop

def top10_read_stats(net,lay,path,tb):
    """
    filename_image_indices = path + net + '_stat_' + lay + '_t10_i.npy'
    @ Args:
        - net (str): network name, e.g., 'n01'
        - lay (str): layer name, e.g., 'conv2'
        - path (str): path to stat files, e.g., '/home/wyeth/w/y/python/pytorch/vis/'
        - tb (str): 't' or 'b' for top or bottom
    """
    fname = path + net + '_' + lay + '_' + tb + '10_'
    ti = np.load(fname + 'i.npy')  # image indices
    tr = np.load(fname + 'r.npy')  # responses
    with open(   fname + 'c.txt',"rb") as fp:   # grid coordinates
        tc = pickle.load(fp)
    return ti, tr, tc

def im5k_get_pltshow(k):
    fname = "/data/images_npy/" + str(k) + ".npy"
    d = np.load(fname)
    print("  Min value:", d.min())
    print("  Max value:", d.max())
    d = d - d.min()
    d = d / d.max()
    im = np.transpose(d, (1,2,0))
    return im

def show_img_and_grad(k,ggrads):
    plt.figure();
    plt.subplot(1,2,1)
    plt.imshow(im5k_get_pltshow(k))  # Plot the original image

    gim = ggrads - ggrads.min()   # Normalize the gradient image
    gim = gim /gim.max()
    gim = np.moveaxis(gim,0,-1)  # Move 1st array dimension to end
    plt.subplot(1,2,2);
    plt.imshow(gim)
    vis.savefig(f"temp{i}.png") # TODO: very temporary solution

name_net = 'n01'
name_lay = 'conv2'
stat_path = '/home/wyeth/w/python/pytorch/std/tb10/'
ti, tr, tc = top10_read_stats(name_net,name_lay,stat_path,'t')  # Top 10
bi, br, bc = top10_read_stats(name_net,name_lay,stat_path,'b')  # Bottom 10

mod = mdl.get_alexnet()
GBP = GuidedBackprop(mod)

targ_layer = 3
nstim = 1
xn = 227
unit = 2

for i in range(10):
  
  d = np.empty((nstim,3,xn,xn), dtype='float32')  # Empty array to hold images
  ii = ti[unit][i]
  fname = "/dataloc/images_npy/" + str(ii) + ".npy"
  d[0] = np.load(fname)
  tt = torch.tensor(d)   # Convert to tensor format
  tt_var = Variable(tt, requires_grad=True)
  guided_grads = GBP.generate_gradients(tt_var, targ_layer, unit)
  show_img_and_grad(ii,guided_grads)