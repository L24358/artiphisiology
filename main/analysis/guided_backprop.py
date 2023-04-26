"""
@ References:
    - https://leslietj.github.io/2020/07/22/Deep-Learning-Guided-BackPropagation/

@ TODO:
    - Use Dataset Imagenette
    - Store result somewhere
    - Visualize top 10, bottom 10
"""
import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import handytools.visualizer as vis
import handytools.navigator as nav
import spectools.basics as bcs
import spectools.models.models as mdl
from spectools.models.gbackprop import Guided_backprop
from spectools.stimulus.dataloader import Imagenette

image = Image.open(os.path.join(nav.datapath, "imagenette", "train", "n01440764", "n01440764_11668.JPEG")).convert('RGB') 
model = models.alexnet(pretrained=True)
model = mdl.get_vgg16()
guided_bp = Guided_backprop(model)

transform = transforms.Compose([
    transforms.Resize(227),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_gbackprop(image):
    X = image.unsqueeze(0).requires_grad_() #transform(image)
    result = guided_bp.visualize(X, None)
    result = bcs.normalize(result)
    return result

dataset = Imagenette(transform = transform)

for i in range(len(dataset)):
    image = dataset[i]
    result = get_gbackprop(image)
    import pdb; pdb.set_trace()


result = get_gbackprop(image)
# import pdb; pdb.set_trace()
# plt.imshow(result)
# vis.savefig()