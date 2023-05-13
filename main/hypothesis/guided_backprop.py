"""
@ References:
    - https://leslietj.github.io/2020/07/22/Deep-Learning-Guided-BackPropagation/

@ TODO:
    - (O) Use Dataset Imagenette
    - (O) Modify model s.t. it works with gbackprop
    - Get receptive field
    - Store (entire) result somewhere
    - Visualize top 10, bottom 10
"""
import gc
import handytools.navigator as nav
import spectools.basics as bcs
import spectools.models.models as mdl
from torchvision import transforms
from spectools.models.gbackprop import Guided_backprop
from spectools.stimulus.dataloader import Imagenette

# hyperparameters
hkey = 11
model_name = "VGG16"

# define transform
transform = transforms.Compose([
    transforms.Resize(227),
    transforms.CenterCrop(227),
    # transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# get gbackprop
def get_gbackprop(image):
    X = image.unsqueeze(0).requires_grad_()
    result = guided_bp.visualize(X, None, model_kwargs={"premature_quit": True})
    result = bcs.normalize(result)
    return result

# get dataset and model
dataset = Imagenette(transform = transform)
model = mdl.get_vgg16(hidden_keys=[hkey])
guided_bp = Guided_backprop(model)

# main
for i in range(len(dataset)):
    image = dataset[i][0]
    result = get_gbackprop(image)
    nav.npsave(result.detach().numpy(), nav.resultpath, f"gbackprop_{model_name}", f"hkey={hkey}_iidx={i}.npy")

    del image, result
    gc.collect()