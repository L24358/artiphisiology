import handytools.navigator as nav
from spectools.models.models import get_parameters

params = get_parameters("vgg16")
nav.pklsave(params, nav.modelpath, "params", "vgg16_parameters.pkl")