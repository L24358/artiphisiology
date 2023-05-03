"""
@ Source:
    https://github.com/csgwon/pytorch-deconvnet/blob/master/models/vgg16_deconv.py
"""
import torch
import numpy as np
import torch.nn as nn
from .models import get_vgg16

class VGG16_deconv(torch.nn.Module):
    def __init__(self):
        super(VGG16_deconv, self).__init__()
        self.conv2DeconvIdx = {0:12, 3:10, 6:8, 8:7, 11:5, 13:4, 16:2, 18:1}
        self.conv2DeconvBiasIdx = {0:10, 3:8, 6:7, 8:5, 11:4, 13:2, 16:1, 18:0}
        self.unpool2PoolIdx = {11:2, 9:5, 6:10, 3:15, 0:20}

        self.deconv_features = torch.nn.Sequential(
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(512, 512, 3, padding=1),
            torch.nn.ConvTranspose2d(512, 512, 3, padding=1),
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(512, 512, 3, padding=1),
            torch.nn.ConvTranspose2d(512, 256, 3, padding=1), # layer 5
            torch.nn.MaxUnpool2d(2, stride=2), # layer 6
            torch.nn.ConvTranspose2d(256, 256, 3, padding=1),
            torch.nn.ConvTranspose2d(256, 128, 3, padding=1),
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(128, 64, 3, padding=1),
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(64, 3, 3, padding=1),
        )

        self.deconv_first_layers = torch.nn.ModuleList([
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(512, 512, 3, padding=1),
            torch.nn.ConvTranspose2d(512, 512, 3, padding=1),
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(512, 512, 3, padding=1),
            torch.nn.ConvTranspose2d(512, 256, 3, padding=1),
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(256, 256, 3, padding=1),
            torch.nn.ConvTranspose2d(256, 128, 3, padding=1),
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(128, 64, 3, padding=1),
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(64, 3, 3, padding=1)]
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # initializing weights using ImageNet-trained model from PyTorch
        vgg16_pretrained = get_vgg16()
        for i, layer in enumerate(vgg16_pretrained.features):
            if isinstance(layer, torch.nn.Conv2d):
                self.deconv_features[self.conv2DeconvIdx[i]].weight.data = layer.weight.data
                biasIdx = self.conv2DeconvBiasIdx[i]
                if biasIdx > 0:
                    self.deconv_features[biasIdx].bias.data = layer.bias.data
        del vgg16_pretrained
    
    def forward(self, x, layer_number, map_number, pool_indices, output_size, batch=0):
        """
        @ Args:
        - map number: filter selected
            - pool_indices: the pool indices that are returned by MaxPool2d
        """
        # must start with a conv2d layer
        start_idx = self.conv2DeconvIdx[layer_number]
        if not isinstance(self.deconv_first_layers[start_idx], torch.nn.ConvTranspose2d):
            raise ValueError('Layer '+str(layer_number)+' is not of type Conv2d')
        
        # set weight and bias
        self.deconv_first_layers[start_idx].weight.data = self.deconv_features[start_idx].weight[map_number].data[None, :, :, :]
        self.deconv_first_layers[start_idx].bias.data = self.deconv_features[start_idx].bias.data

        # first layer will be single channeled, since we're picking a particular filter
        output = self.deconv_first_layers[start_idx](x)

        # transpose conv through the rest of the network
        for i in range(start_idx+1, len(self.deconv_features)):
            if isinstance(self.deconv_features[i], torch.nn.MaxUnpool2d):
                output = self.deconv_features[i](
                    output,
                    pool_indices[self.unpool2PoolIdx[i]][batch],
                    output_size[self.unpool2PoolIdx[i]-1][batch],
                )
            else:
                output = self.deconv_features[i](output)
        return output
    
class GuidedBackprop():
  """
     Produces gradients generated with guided back propagation from given image
  """
  def __init__(self, model):
    self.model = model
    self.gradients = None
    self.outputs = []
    self.forward_relu_outputs = []
    # Put model in evaluation mode
    self.model.eval()
    self.update_relus()
    self.hook_layers()
  
  def hook_layers(self):
    def hook_function(module, grad_in, grad_out):
        self.gradients = grad_in[0]
    # Register hook to the first layer
    first_layer = list(self.model.features._modules.items())[0][1]
    first_layer.register_backward_hook(hook_function)
  
  def update_relus(self):
    """
       Updates relu activation functions so that
         1- stores output in forward pass
         2- imputes zero for gradient values that are less than zero
    """
    def relu_backward_hook_function(module, grad_in, grad_out):
      """
      If there is a negative gradient, change it to zero
      """
      # Get last forward output
      last_forward_output = self.forward_relu_outputs[-1]
      last_forward_output[last_forward_output > 0] = 1
      modified_grad_out = last_forward_output * torch.clamp(grad_in[0], min=0.0)
      del self.forward_relu_outputs[-1]  # Remove last forward output
      return (modified_grad_out,)
    
    def relu_forward_hook_function(module, ten_in, ten_out):
      """
      Store results of forward pass
      """
      self.forward_relu_outputs.append(ten_out)
    
    # Loop through layers, hook up ReLUs
    for pos, module in self.model.features._modules.items():
      if isinstance(module, nn.ReLU):
        module.register_backward_hook(relu_backward_hook_function)
        module.register_forward_hook(relu_forward_hook_function)
  
  def generate_gradients(self, input_image, target_layer, target_filter):
    self.model.zero_grad()
    # Forward pass
    x = input_image
    for index, layer in enumerate(self.model.features):
      # Forward pass layer by layer
      # x is not used after this point because it is only needed to trigger
      # the forward hook function
      x = layer(x)
      # Only need to forward until the selected layer is reached
      if index == target_layer:
        # (forward hook function triggered)
        break
    self.outputs = x;
    
    # Target for backprop - find max response from target layer, target filter
    pos = np.where(x[0,target_filter,:,:]==x[0,target_filter,:,:].max())[:];
    one_hot_output = torch.FloatTensor(x.shape).zero_()
    one_hot_output[0,target_filter,pos[0][0],
                   pos[1][0]] = x[0,target_filter,pos[0][0],pos[1][0]]
    # Backward pass
    x.backward(gradient=one_hot_output)
    
    # Convert Pytorch variable to numpy array
    # [0] to get rid of the first channel (1,3,224,224)
    gradients_as_arr = self.gradients.data.numpy()[0]
    return gradients_as_arr
  
  def generate_grads_min(self, input_image, target_layer, target_filter):
    self.model.zero_grad()
    # Forward pass
    x = input_image
    for index, layer in enumerate(self.model.features):
      # Forward pass layer by layer
      # x is not used after this point because it is only needed to trigger
      # the forward hook function
      x = layer(x)
      # Only need to forward until the selected layer is reached
      if index == target_layer:
        # (forward hook function triggered)
        break
    self.outputs = x;
    
    # Target for backprop - find max response from target layer, target filter
    pos = np.where(x[0,target_filter,:,:]==x[0,target_filter,:,:].min())[:];
    one_hot_output = torch.FloatTensor(x.shape).zero_()
    one_hot_output[0,target_filter,pos[0][0],
                   pos[1][0]] = x[0,target_filter,pos[0][0],pos[1][0]]
    # Backward pass
    x.backward(gradient=one_hot_output)
    
    # Convert Pytorch variable to numpy array
    # [0] to get rid of the first channel (1,3,224,224)
    gradients_as_arr = self.gradients.data.numpy()[0]
    return gradients_as_arr