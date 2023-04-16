import torch
import torch.nn as nn
import numpy as np
import handytools.navigator as nav
# from .resnet import ResNet

def get_parameters(name):
    if name == "alexnet":
        net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    elif name == "vgg16":
        net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
    elif name == "resnet18":
        net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    else:
        raise ValueError(f"`name` cannot be {name}.")
    params = {name: parameter for name, parameter in net.named_parameters()}
    return params

def get_alexnet(hidden_keys=[]):
    params = nav.pklload("/src", "data", "models", "alexnet_parameters.pkl")
    model = AlexNet(hidden_keys=hidden_keys)
    model.load_state_dict(params)
    return model

def get_vgg16(hidden_keys=[]):
    params = nav.pklload("/src", "data", "models", "vgg16_parameters.pkl")
    model = VGG16(hidden_keys=hidden_keys)
    model.load_state_dict(params)
    return model

# def get_resnet18(hidden_keys=[]):
#     params = nav.pklload("/src", "data", "models", "resnet18_parameters.pkl")
#     model = ResNet(hidden_keys=hidden_keys)
#     model.load_state_dict(params)
#     return model

class VGG16(nn.Module):
    def __init__(
        self, hidden_keys: list = [], num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        # initialize hidden info
        self.hidden_info = {}
        self.update_hidden_keys(hidden_keys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()

        # features
        for i, child in enumerate(list(self.features.children())):
            x = child(x)
            if i in self.hidden_keys: self.hidden_info[i].append(x)

        # avg pooling and flatten
        x = self.avgpool(x)
        if i+1 in self.hidden_keys: self.hidden_info[i+1].append(x) # layer i+1
        x = torch.flatten(x, 1)

        # classifier
        for j, child in enumerate(list(self.classifier.children())):
            x = child(x)
            if (i+1)+(j+1) in self.hidden_keys: self.hidden_info[i+j+2].append(x) # starts from layer i+2=(i+1)+(j+1) because j=0
        return x

    def update_hidden_keys(self, keys):
        dic = {}
        self.hidden_keys = keys
        for key in self.hidden_keys: dic[key] = []
        self.hidden_info = {**dic, **self.hidden_info} # update, with priority given to existing self.hidden_info

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
        for j, child in enumerate(list(self.classifier.children())):
            x = child(x)
            if i+(j+1) in self.hidden_keys: self.hidden_info[i+j+1].append(x)
        return x

    def update_hidden_keys(self, keys):
        dic = {}
        self.hidden_keys = keys
        for key in self.hidden_keys: dic[key] = []
        self.hidden_info = {**dic, **self.hidden_info} # update, with priority given to existing self.hidden_info

if __name__ == "__main__":
    get_parameters("vgg16")