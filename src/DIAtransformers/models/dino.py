from cgitb import reset
import torch
from torch import nn, tensor
"""
Implementation of META's DINO model
"""


"""
DINO model 

Backbone (for preprocessing the inputs into a "pallatable" format)
    -  The backbone of DINO is a resNet model in the original paper.
    - There is also experimentation with the Swin transformer.
"""
resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)


"""
Positional embeddings:
    -  In the paper, they use a sinusoidal embedding similar to the original transformer paper
       (Vaswani et al., 2017)

Attention: 
    - For the encoder, the attention layers are the same as they were in the original transformer (Vaswani et al., 2017)
"""

from .utils import multiHeadAttention

"""

"""

"""
Encoder:
They use a traditional transformer encoder (Vaswani et al., 2017)
"""
class dinoEncoder(nn.Module):
    def __init__(self, num_heads, dim, n):
        super(dinoEncoder, self).__init__()
        # number of attention heads
        self.num_heads = num_heads
        # number of attention blocks
        self.dim = dim
        # layer normalization, to stabilize the gradients
        # should not depend on batch size
        self.norm = nn.LayerNorm(normalized_shape = (n, dim), elementwise_affine = True)
        self.attention = multiHeadAttention(self.num_heads, dim, n)
        self.dh = int(self.dim / self.num_heads)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, input):
        whoa = self.norm(input)
        uhOh = self.attention.forward(whoa)
        uhHuh = self.norm(uhOh + input)
        toAdd = uhOh + input    
        output = self.mlp(uhHuh)
        output += toAdd
        return output


"""
Decoder:
"""
class dino(nn.Module):
    def __init__(self, dim, height = 224, width = 224, patch_res = 16):
        # In the paper they use 
        self.featureExtractor = resnet
        self.n = int((height * width) / (patch_res ** 2))
        self.encoder = dinoEncoder(8, dim)
        pass
