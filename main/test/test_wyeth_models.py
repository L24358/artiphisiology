import numpy as np
import spectools.models.models as mdl
from torchvision import models
from torchvision.models import AlexNet_Weights

mod1 = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
mod2 = mdl.get_alexnet(hidden_keys=[2,3])

dic = {}
dic2 = {}
for n, p in mod1.named_parameters(): dic[n] = p.data
for n, p in mod2.named_parameters(): dic2[n] = p.data

for key in dic.keys():
    arr1 = dic[key].numpy().flatten()
    arr2 = dic2[key].numpy().flatten()
    print(np.all(arr1 == arr2))