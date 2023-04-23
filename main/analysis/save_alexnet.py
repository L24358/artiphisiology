import handytools.navigator as nav
from spectools.models.models import get_parameters

params = get_parameters("alexnet")
nav.pklsave(params, nav.modelpath, "params", "alexnet_parameters.pkl")