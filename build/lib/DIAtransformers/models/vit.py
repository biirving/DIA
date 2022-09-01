
import numpy as np
from pandas import array
import torch
from torch import BFloat16Storage, Tensor, bfloat16, nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import PIL
import sys, os 


class multiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim, n):
        super(multiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.n = n
        self.Dh = int(self.dim/self.num_heads)

        self.softmax = nn.Softmax(dim = -1)
        # The matrix which multiplies all of the attention heads at the end
        self.multi_mad = nn.Linear(self.num_heads * 3 * self.Dh, self.dim)

        # these weights will be initialized randomly
        # in terms of the weights, they will eventually attend to different parts of the inputs in a similar way
        self.q = nn.Linear(self.dim, 3 * self.Dh * self.num_heads)
        self.v = nn.Linear(self.dim, 3 * self.Dh * self.num_heads)
        self.k = nn.Linear(self.dim, 3 * self.Dh * self.num_heads)
        
    def forward(self, input):
        # q, k, v matrices
        q_mat = rearrange(self.q(input), 'b n (h d) -> b h n d', h = self.num_heads)
        v_mat = rearrange(self.k(input), 'b n (h d) -> b h n d', h = self.num_heads)
        k_mat = rearrange(self.v(input), 'b n (h d) -> b h n d', h = self.num_heads)

        # Softmax step, calculated for each row of each head
        inter = self.softmax(torch.matmul(q_mat, torch.transpose(k_mat, 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))

        # prepare the vector for input
        final = rearrange(torch.matmul(inter, v_mat), 'b h n d -> b n (h d)', h = self.num_heads)

        # final computation
        return self.multi_mad(final)


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




# You should be able to feed an input of any batch size to the model
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
        self.posEmbed = torch.expand(self.pos_embed, self.n + 1, dim)


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

