import handytools.navigator as nav
from spectools.models.models import get_parameters

params = get_parameters("vgg16")
nav.pklsave(params, "/src", "data", "models", "vgg16_parameters.pkl")