import os
import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
import handytools.visualizer as vis

exec(open(nav.homepath + "data/wyeth_foi/d01_net.py").read())
exec(open(nav.homepath + "data/wyeth_foi/d05_shape_util.py").read())

# get dictionary for stimulus
xn = 227
sz = 50 # to be determined
lw = 1
fg = 1.0
bg = 0.0
splist = stimset_dict_shape_fo_1(xn,sz,lw,fg,bg) # List of dictionary entries, one per stimulus image

fillflag = 1 # hollow
for d in splist[1:]:
  image_array = stimset_stim_get_shape_fo(d,fillflag)
  plt.imshow(image_array)
  vis.savefig()
  diameter = max(image_array.sum(axis=1)/1.)

  import pdb; pdb.set_trace()