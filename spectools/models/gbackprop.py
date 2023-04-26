import torch
import torch
from torch import nn
from torchvision import models, transforms

class Guided_backprop():
    """
    Source: https://leslietj.github.io/2020/07/22/Deep-Learning-Guided-BackPropagation/
    """
    def __init__(self, model):
        self.model = model
        self.image_reconstruction = None # store R0
        self.activation_maps = []  # store f1, f2, ... 
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            self.image_reconstruction = grad_in[0] 

        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop() 
            # for the forward pass, after the ReLU operation, 
            # if the output value is positive, we set the value to 1,
            # and if the output value is negative, we set it to 0.
            grad[grad > 0] = 1 
            
            # grad_out[0] stores the gradients for each feature map,
            # and we only retain the positive gradients
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad

            return (new_grad_in,)


        # AlexNet model 
        modules = list(self.model.features.named_children())

        # travese the modules，register forward hook & backward hook
        # for the ReLU
        for name, module in modules:
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)

        # register backward hook for the first conv layer
        first_layer = modules[0][1] 
        first_layer.register_backward_hook(first_layer_hook_fn)

    def visualize(self, input_image, target_class, model_kwargs = {}):
        model_output = self.model(input_image, **model_kwargs)
        self.model.zero_grad()
        pred_class = model_output.argmax().item()
        
        # grad_target_map = torch.zeros(model_output.shape,
        #                               dtype=torch.float)
        # if target_class is not None:
        #     grad_target_map[0][target_class] = 1
        # else:
        #     grad_target_map[0][pred_class] = 1
        # import pdb; pdb.set_trace()
        
        model_output.backward(model_output) # grad_target_map
        
        result = self.image_reconstruction.data[0].permute(1,2,0)
        return result.numpy()

