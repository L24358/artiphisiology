"""
Conv2d: 
"""

import spectools.models.models as mdl
from spectools.models.calc import get_RF_wrap

mod = mdl.get_alexnet()
dic = get_RF_wrap(mod)

import pdb; pdb.set_trace()