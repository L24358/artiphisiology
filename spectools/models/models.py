import torch
import torch.nn as nn
import numpy as np
import handytools.navigator as nav
from spectools.models.resnet import ResNet, BasicBlock

AN_layer = {0: "Conv1", 3: "Conv2", 6: "Conv3", 8: "Conv4", 10: "Conv5", 15: "FC1", 18: "FC2", 20: "FC3"}
VGG16_layer = {0: "Conv1", 3: "Conv2", 6: "Conv3", 8: "Conv4", 11: "Conv5", 13: "Conv6", 16: "Conv7", 18: "Conv8", 22: "FC1", 25: "FC2", 28: "FC3"}
VGG16b_layer = {0: "Conv1", 2: "Conv2", 5: "Conv3", 7: "Conv4", 10: "Conv5", 12: "Conv6", 14: "Conv7", 17: "Conv8", 19: "Conv9", 21: "Conv10", 24: "Conv11", 26: "Conv12", 28: "Conv13", 32: "FC1", 35: "FC2", 38: "FC3"}
ResNet18_layer = {0: "Conv1", 4: "b-Block1.1", 5: "b-Block1.2", 6: "b-Block2.1", 7: "b-Block2.2",
                  8: "b-Block3.1", 9: "b-Block3.2", 10: "b-Block4.1", 11: "b-Block4.2"}
AN_units = {3: 192, 6: 384, 8: 256}
ResNet18_units = {0: 64, 4: 64, 5: 64, 6: 128, 7: 128, 8: 256, 9: 256, 10: 512, 11: 512}

def get_units(mtype, hkey):
    if mtype == "AN": return AN_units[hkey]
    elif mtype == "ResNet18": return ResNet18_units[hkey]

def get_names(mtype, hkey):
    if mtype == "AN": return AN_layer[hkey]
    elif mtype == "VGG16b": return VGG16b_layer[hkey]
    elif mtype == "ResNet18": return ResNet18_layer[hkey]

def get_parameters(name):
    if name == "alexnet":
        net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True) # "AlexNet_Weights.IMAGENET1K_V1"
    elif name == "vgg16":
        net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True) # weights="s"
    elif name == "vgg16b":
        net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    elif name == "resnet18":
        net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    else:
        raise ValueError(f"`name` cannot be {name}.")
    params = {name: parameter for name, parameter in net.named_parameters()}
    return params

def get_additional_resnet18():
    net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    layers = {net.bn1: "bn1",
              net.layer1[0].bn1: "layer1.0.bn1", net.layer1[0].bn2: "layer1.0.bn2",
              net.layer1[1].bn1: "layer1.1.bn1", net.layer1[1].bn2: "layer1.1.bn2",
              net.layer2[0].bn1: "layer2.0.bn1", net.layer2[0].bn2: "layer2.0.bn2",
              net.layer2[1].bn1: "layer2.1.bn1", net.layer2[1].bn2: "layer2.1.bn2",
              net.layer3[0].bn1: "layer3.0.bn1", net.layer3[0].bn2: "layer3.0.bn2",
              net.layer3[1].bn1: "layer3.1.bn1", net.layer3[1].bn2: "layer3.1.bn2",
              net.layer4[0].bn1: "layer4.0.bn1", net.layer4[0].bn2: "layer4.0.bn2",
              net.layer4[1].bn1: "layer4.1.bn1", net.layer4[1].bn2: "layer4.1.bn2",
              net.layer2[0].downsample[1]: "layer2.0.downsample.1",
              net.layer3[0].downsample[1]: "layer3.0.downsample.1",
              net.layer4[0].downsample[1]: "layer4.0.downsample.1",
    }

    params = {}
    for layer in layers:
        name = layers[layer]
        params[name + ".running_mean"] = layer.running_mean
        params[name + ".running_var"] = layer.running_var
    return params

def get_alexnet(hidden_keys=[], in_place=False):
    params = nav.pklload(nav.modelpath, "params", "alexnet_parameters.pkl")
    model = AlexNet(hidden_keys=hidden_keys, in_place=in_place)
    model.load_state_dict(params)
    return model

def get_vgg16(hidden_keys=[]):
    params = nav.pklload(nav.modelpath, "params", "vgg16_parameters.pkl")
    model = VGG16(hidden_keys=hidden_keys)
    model.load_state_dict(params)
    return model

def get_vgg16b(hidden_keys=[]):
    params = nav.pklload(nav.modelpath, "params", "vgg16b_parameters.pkl")
    model = VGG16b(hidden_keys=hidden_keys)
    model.load_state_dict(params)
    return model

def get_resnet18(hidden_keys=[]):
    params = nav.pklload(nav.modelpath, "params", "resnet18_parameters.pkl")
    params2 = nav.pklload(nav.modelpath, "params", "resnet18_parameters_add.pkl")
    params.update(params2)
    model = ResNet(BasicBlock, [2, 2, 2, 2], hidden_keys=hidden_keys)
    model.load_state_dict(params)
    return model

def load_model(mtype, hkeys, device):
    if mtype == "AN": mfunc = get_alexnet
    elif mtype == "VGG16": mfunc = get_vgg16
    elif mtype == "VGG16b": mfunc = get_vgg16b
    elif mtype == "ResNet18": mfunc = get_resnet18
    model = mfunc(hidden_keys=hkeys).to(device)
    return model

class VGG16(nn.Module):
    def __init__(
        self, hidden_keys: list = [], num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5,) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # layer 11
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(False),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(False),
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
        self.hidden_keys = hidden_keys
        self.reset_storage()
        self.flag_output = True

    def forward(self, x: torch.Tensor, premature_quit=True, filt=lambda x:x) -> torch.Tensor:
        x = x.float()

        # features
        for i, child in enumerate(list(self.features.children())):
            if not isinstance(child, nn.MaxPool2d): 
                x = child(x)
            else: # for Maxpool layers
                x, maxpoolidx = child(x)
                self.pool_indices[i].append(maxpoolidx)

            if i in self.hidden_keys:
                self.hidden_info[i].append(filt(x))

                if premature_quit and i == max(self.hidden_keys):
                    self.flag_output = False # only need to append once
                    return x
            
            if self.flag_output:
                self.output_size[i].append(x.shape)
        self.flag_output = False # only need to append once
                
        # avg pooling and flatten
        x = self.avgpool(x)
        if i+1 in self.hidden_keys: self.hidden_info[i+1].append(filt(x)) # layer i+1
        x = torch.flatten(x, 1)

        # classifier
        for j, child in enumerate(list(self.classifier.children())):
            x = child(x)
            if (i+1)+(j+1) in self.hidden_keys: 
                self.hidden_info[i+j+2].append(filt(x)) # starts from layer i+2=(i+1)+(j+1) because j=0
        return x

    def update_hidden_keys(self):
        dic = {}
        for key in self.hidden_keys: dic[key] = []
        self.hidden_info = {**dic, **self.hidden_info} # update, with priority given to existing self.hidden_info

    def reset_storage(self):
        self.hidden_info = {}
        self.update_hidden_keys()
        self.pool_indices = {}
        for layer in [2, 5, 10, 15, 20]: self.pool_indices[layer] = []
        self.output_size = {}
        for layer in range(21): self.output_size[layer] = []

class VGG16b(nn.Module):
    def __init__(
        self, hidden_keys: list = [], num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5, in_place = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=in_place),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=in_place),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=in_place),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True),
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
        self.hidden_keys = hidden_keys
        self.hidden_info = {}
        self.reset_storage()

    def forward(self, x: torch.Tensor, premature_quit=True, filt=lambda x:x) -> torch.Tensor:
        x = x.float()

        # features
        for i, child in enumerate(list(self.features.children())):
            x = child(x)

            if i in self.hidden_keys:
                self.hidden_info[i].append(filt(x))

                if premature_quit and i == max(self.hidden_keys):
                    self.flag_output = False # only need to append once
                    return x
                
        # avg pooling and flatten
        x = self.avgpool(x)
        if i+1 in self.hidden_keys: self.hidden_info[i+1].append(filt(x)) # layer i+1
        x = torch.flatten(x, 1)

        # classifier
        for j, child in enumerate(list(self.classifier.children())):
            x = child(x)
            if (i+1)+(j+1) in self.hidden_keys:
                self.hidden_info[i+j+2].append(filt(x)) # starts from layer i+2=(i+1)+(j+1) because j=0
                
        return x

    def update_hidden_keys(self):
        dic = {}
        for key in self.hidden_keys: dic[key] = []
        self.hidden_info = {**dic, **self.hidden_info} # update, with priority given to existing self.hidden_info

    def reset_storage(self):
        del self.hidden_info
        self.hidden_info = {}
        self.update_hidden_keys()

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, hidden_keys=[], in_place=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=in_place),
            nn.MaxPool2d(kernel_size = 3, stride = 2, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=in_place),
            nn.MaxPool2d(kernel_size = 3, stride = 2, dilation=1, ceil_mode=False),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=in_place),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=in_place),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=in_place),
            nn.MaxPool2d(kernel_size = 3, stride = 2, dilation=1, ceil_mode=False),

            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5, inplace=False),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=in_place),

            nn.Dropout(0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=in_place),

            nn.Linear(4096, num_classes)
        )

        # initialize hidden info
        self.hidden_info = {}
        self.update_hidden_keys(hidden_keys)
        
    def forward(self, x, premature_quit=True):
        x = x.float()
        for i, child in enumerate(list(self.features.children())):
            x = child(x)
            if i in self.hidden_keys:
                self.hidden_info[i].append(x)
                if premature_quit and i == max(self.hidden_keys): return x
        for j, child in enumerate(list(self.classifier.children())):
            x = child(x)
            if i+(j+1) in self.hidden_keys: self.hidden_info[i+j+1].append(x)
        return x

    def update_hidden_keys(self, keys):
        dic = {}
        self.hidden_keys = keys
        for key in self.hidden_keys: dic[key] = []
        self.hidden_info = {**dic, **self.hidden_info} # update, with priority given to existing self.hidden_info

    def reset_storage(self):
        del self.hidden_info
        self.hidden_info = {}
        self.update_hidden_keys(self.hidden_keys)

if __name__ == "__main__":
    if False:
        params = get_parameters("vgg16b")
        nav.pklsave(params, nav.modelpath, "params", "vgg16b_parameters.pkl")

    net = get_vgg16b()