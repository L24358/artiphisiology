import torch
import spectools.models.models as mdl

net = mdl.get_resnet18(hidden_keys=[0])
X = torch.rand((1,3,227,227))
y = net(X)