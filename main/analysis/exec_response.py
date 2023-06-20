"""
Example code for how to get response of candidate network for a particular stimulus desired.

``get_response_wrapper`` returns a dictionary with hidden layer indices as key, and center responses (shape=(#units, batch size)) as values.
"""

import spectools.models.models as mdl
from spectools.stimulus.wyeth import get_stimulus
from spectools.responses import get_response_wrapper

# params
xn = 227
sz = 50
lw = 1.5
fg = 1.0
bg = 0.0
hkeys = list(mdl.VGG16b_layer.keys()) # the key for hidden layers
mtype = "VGG16b"

fill = get_stimulus(1, xn=xn, sz=sz, lw=lw, fg=fg, bg=bg) # get filled stimulus
fname = lambda hkey: f"hkey={hkey}_fill=1_xn={xn}_sz={sz}_lw={lw}_fg={fg}_bg={bg}.npy" # specify the format in which files will be saved as
R_fill = get_response_wrapper(hkeys, fill, fname, mtype=mtype) # get the center response of the model for different layers (hkeys)

outline = get_stimulus(0, xn=xn, sz=sz, lw=lw, fg=fg, bg=bg) # get hollow stimulus
fname = lambda hkey: f"hkey={hkey}_fill=0_xn={xn}_sz={sz}_lw={lw}_fg={fg}_bg={bg}.npy"
R_outline = get_response_wrapper(hkeys, outline, fname, mtype=mtype)
