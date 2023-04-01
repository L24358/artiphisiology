import torch
import torch.nn as nn
import numpy as np

def get_parameters(name):
    if name == "alexnet":
        net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    else:
        raise ValueError(f"`name` cannot be {name}.")
    params = {name: parameter for name, parameter in net.named_parameters()}
    return params

def get_alexnet(hidden_keys=[]):
    params = get_parameters("alexnet")
    model = AlexNet(hidden_keys=hidden_keys)
    model.load_state_dict(params)
    return model

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, hidden_keys=[]):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, dilation=1, ceil_mode=False),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, dilation=1, ceil_mode=False),

            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5, inplace=False),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

        # initialize hidden info
        self.hidden_info = {}
        self.update_hidden_keys(hidden_keys)
        
    def forward(self, x):
        x = x.float()
        for i, child in enumerate(list(self.features.children())):
            x = child(x)
            if i in self.hidden_keys: self.hidden_info[i].append(x)
        out = self.classifier(x)
        return out

    def update_hidden_keys(self, keys):
        dic = {}
        self.hidden_keys = keys
        for key in self.hidden_keys: dic[key] = []
        self.hidden_info = {**dic, **self.hidden_info} # update, with priority given to existing self.hidden_info

