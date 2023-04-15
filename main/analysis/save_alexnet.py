import handytools.navigator as nav
from spectools.models.models import get_parameters

params = get_parameters("alexnet")
nav.pklsave(params, "/src", "data", "models", "alexnet_parameters.pkl")