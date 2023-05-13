"""
Example code for how to get response of candidate network for a particular stimulus desired.
"""

from spectools.stimulus.wyeth import get_stimulus
from spectools.responses import get_response_wrapper

# params
xn = 227
sz = 50
lw = 1.5
fg = 1.0
bg = 0.0
hkeys = [0, 3, 6, 8, 10]
mtype = "AN"

fill = get_stimulus(1, xn=xn, sz=sz, lw=lw, fg=fg, bg=bg)
fname = lambda hkey: f"hkey={hkey}_fill=1_xn={xn}_sz={sz}_lw={lw}_fg={fg}_bg={bg}.npy"
R_fill = get_response_wrapper(hkeys, fill, fname, mtype=mtype)

outline = get_stimulus(0, xn=xn, sz=sz, lw=lw, fg=fg, bg=bg)
fname = lambda hkey: f"hkey={hkey}_fill=0_xn={xn}_sz={sz}_lw={lw}_fg={fg}_bg={bg}.npy"
R_outline = get_response_wrapper(hkeys, outline, fname, mtype=mtype)
