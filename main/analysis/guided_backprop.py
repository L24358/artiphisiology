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
from spectools.models.gbackprop import Guided_backprop

image = Image.open(os.path.join(nav.datapath, "imagenette", "train", "n01440764", "n01440764_11668.JPEG")).convert('RGB') 
model = models.alexnet(pretrained=True)
guided_bp = Guided_backprop(model)
print('AlexNet Architecture:\n', '-'*60, '\n', model, '\n', '-'*60)


transform = transforms.Compose([
    transforms.Resize(227),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
X = transform(image).unsqueeze(0).requires_grad_()
result = guided_bp.visualize(X, None)
result = bcs.normalize(result)
plt.imshow(result)
vis.savefig()