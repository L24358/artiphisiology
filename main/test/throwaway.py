import numpy as np
import matplotlib.pyplot as plt
import handytools.navigator as nav
from spectools.old.metrics import fvmax, cubic_spline

shape_coor = nav.pklload("/src", "data", "stimulus", "shape_coor.pkl")
ps, dps, ddps = cubic_spline(np.array(shape_coor[0]).reshape(-1, 2))

#================

from scipy.interpolate import CubicSpline

def cubic_spline2(invec, ival=1/200): # invec.shape = (-1, 2)
    cs = CubicSpline(np.linspace(0, 1, len(invec)), invec)
    xs = np.arange(0, 1, ival)
    return cs(xs), cs(xs, 1), cs(xs, 2)

ps, dps, ddps = cubic_spline2(np.array(shape_coor[0]).reshape(-1, 2))

import pdb; pdb.set_trace()