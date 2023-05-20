import torch
import torch.nn as nn
import numpy as np

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create input tensor from np.arange
d = 5
input_tensor = torch.from_numpy(np.arange(d * d).reshape((1, 1, d, d))).float()
input_tensor2 = [input_tensor, input_tensor*2, input_tensor*5]
input_tensor2 = torch.vstack(input_tensor2).swapaxes(0, 1)
input_tensor3 = torch.vstack([input_tensor, input_tensor*5])

bn = nn.BatchNorm2d(1)
bn2 = nn.BatchNorm2d(3)

output1 = bn(input_tensor) # 1 batch, 1 channel (--> mean=0, std=1)
output2 = bn2(input_tensor2) # 1 batch, 3 channels (--> independent across channels)
output3 = bn(input_tensor3)

# Conclusion: is normalizing across space, not channels!
import pdb; pdb.set_trace()