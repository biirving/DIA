
import numpy as np
from pandas import array
import torch
from torch import BFloat16Storage, Tensor, bfloat16, nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import PIL
import sys, os 
from .utils import multiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, num_heads, dim, n):
        super(EncoderBlock, self).__init__()
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
Vision Transformer model
args
    - Height 
        height of the image, the original paper used the 224 x 224 image size
    - Width
        width of the image, the original paper used the 224 x 224 image size
    - patch_res
        resolution of the patch embeddings created by the model, the original paper condenses the information into 16x16 pixel segments
    - dim
        dimension of the embedded images
    - num_classes
        the number of prediction classes
    - batch size
        number of inferences to be made at once
"""
class vit(nn.Module):

    def __init__(self, height, width, patch_res, dim, num_classes, batch_size):
        super(vit, self).__init__()
        self.checkPass = True
        self.height = height
        self.width = width
        self.channels = 3
        self.num_classes = num_classes
        self.patch_res = patch_res 
        self.dim = dim
        self.patch_dim = self.channels * patch_res * patch_res
        self.n = int((height * width) / (patch_res ** 2))

        # the patch embeddings for the vision transformer
        # the 'height','width','channel', and 'batch' are gleaned from the shape of the tensor
        self.patchEmbed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, dim),)

        # the class token serves as a representation of the entire sequence
        self.classtkn = nn.Parameter(torch.randn(batch_size, 1, dim))
        # this will be concated to the end of the input before positional embedding
        # should the class token be randomonly initialized? or the same across batches? See what happens during training

        # the positional embedding should be applied based on what 
        self.pos_embed = nn.Parameter(torch.randn(batch_size, self.n + 1, dim))
        
        self.encoderBlocks = nn.ModuleList([EncoderBlock(num_heads = 8, dim = dim, n = self.n + 1) for i in range(8)])

        
        self.mlpHead = nn.Sequential(nn.LayerNorm(dim), nn.GELU(), nn.Linear(self.dim, num_classes))
        self.dropout = nn.Dropout(0.1)


            

    def forward(self, img):
        input = self.patchEmbed(img)
        input = torch.cat((input, self.classtkn), dim = 1)
        input += self.pos_embed
        input = self.dropout(input)
        for encoder in self.encoderBlocks:
            output = encoder.forward(input)
            input = output
        out = self.mlpHead(output[:, 0])
        return out
        
    # input will always have batchsize included?
    def adjustBatchSize(self, newBatch):
        # We can just expand the class and position tokens according to the batch size
        self.classtkn = torch.expand(self.classtkn, 1, self.dim)
        self.posEmbed = torch.expand(self.pos_embed, self.n + 1, self.dim)


    # alternating sin/cos embeddings from the first paper 
    def applyPositionalEncodings(self, input:Tensor):
        for x in range(self.n):
            if(x % 2 == 0):
                val = math.sin(x / (10000 ** (2 / (self.dim))))
            else:
                val = math.cos(x / (10000 ** (2 / (self.dim))))
            positional = torch.tensor([val] * self.dim)
            input[0][x] += positional
        return input

