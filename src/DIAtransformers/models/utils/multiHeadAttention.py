import numpy as np
from pandas import array
import torch
from torch import BFloat16Storage, Tensor, bfloat16, nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import PIL
import sys, os 

# classic multihead attention
class multiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim, n, mask=False):
        super(multiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.n = int(n)
        self.Dh = int(self.dim/self.num_heads)
        self.mask = mask

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
        # Compute attention scores using dot product of queries and keys
        scores = torch.matmul(q_mat, torch.transpose(k_mat, 2, 3)) / math.sqrt(self.Dh * self.num_heads)
        # only attend to the previous considered positions (unidirectional attention)
        if(self.mask):
            # Create a mask matrix to prevent attending to future positions
            mask = torch.tril(torch.ones(scores.size(-1), scores.size(-1))).unsqueeze(0).unsqueeze(1).to(input.device)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        weights = torch.softmax(scores, dim=-1)
        # Apply attention weights to values
        inter = torch.matmul(weights, v_mat)
        # reshape for the linear layer
        inter = rearrange(inter, 'b h n d -> b n (h d)')
        output = self.multi_mad(inter)
        return output

