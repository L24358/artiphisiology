import handytools.navigator as nav
from spectools.models.models import get_parameters, get_additional_resnet18

model_name = "resnet18" # alexnet, vgg16, resnet18
params = get_parameters(model_name)
nav.pklsave(params, nav.modelpath, "params", f"{model_name}_parameters.pkl")

if model_name == "resnet18":
    add_params = get_additional_resnet18()
    nav.pklsave(add_params, nav.modelpath, "params", f"{model_name}_parameters_add.pkl")