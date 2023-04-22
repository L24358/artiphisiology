import handytools.navigator as nav
from spectools.models.models import get_parameters

params = get_parameters("alexnet")
nav.pklsave(params, nav.datapath, "models", "alexnet_parameters.pkl")