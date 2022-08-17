# an implementation of the timeSformer model

from torch import nn, tensor
import torch
import numpy as np
from einops import rearrange, repeat
import math



# divided space-time attention
class dividedSpaceTimeAttention(nn.Module):
    def __init__(self, num_heads, dim, n, num_frames):
        super(dividedSpaceTimeAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.n = int(n)
        self.Dh = int(self.dim/self.num_heads)

        self.num_frames = num_frames

        # softmax within each head
        self.softmax = nn.Softmax(dim = -1)

        # these weights will be initialized randomly
        # in terms of the weights, they will eventually attend to different parts of the inputs in a similar way
        self.q = nn.Linear(self.dim, 3 * self.Dh * self.num_heads)
        self.v = nn.Linear(self.dim, 3 * self.Dh * self.num_heads)
        self.k = nn.Linear(self.dim, 3 * self.Dh * self.num_heads)




        self.multimad_temp = nn.Linear(self.num_heads * 3 * self.Dh, self.dim) 
       # self.multi_mad_temp = nn.Parameter(nn.init.xavier_uniform(torch.tensor()))

        # The matrix which multiplies all of the attention heads at the end
        # this will change depending on the num frames?
        self.multi_mad_final = nn.Linear(self.num_heads * 3 * self.Dh, self.dim)


    
    def forward(self, input):

            # q, k, v matrices
            q_mat = rearrange(self.q(input), 'b nf (h d) -> b h nf d', h = self.num_heads)
            v_mat = rearrange(self.k(input), 'b nf (h d) -> b h nf d', h = self.num_heads)
            k_mat = rearrange(self.v(input), 'b nf (h d) -> b h nf d', h = self.num_heads)

            # First, we calculate temporal attention, multiplying the q vector being processed by every k vector at that 
            # frame in subsequent time steps

            # at this point, the q matrix contains all of the query vectors to be processed
            # but each row will be multiplied by a different set of the k vectors, because they are being compared to the other
            # keys at that timeframe
            temporal = self.softmax(torch.matmul(q_mat[:, :, 0::self.n, :], torch.transpose(k_mat[:, :, 0::self.n, :], 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
            temporal = torch.matmul(temporal, v_mat[:, :, 0::self.n, :])
            temporal = torch.sum(temporal, 2, keepdim = True)
           # print(temporal.shape)
            temporal = temporal.repeat(1, 1, self.num_frames, 1)
           # print(temporal.shape)
            
            # temporal calculation
            for x in range(1, self.n):
                # get all of the patches at that frame
                #print(torch.transpose(q_mat, 2 , 3).shape)
                #inter = self.softmax(torch.matmul(torch.transpose(q_mat, 2 , 3), k_mat[:, :, x::self.n, :]) / (math.sqrt(self.Dh) * self.num_heads))
                inter = self.softmax(torch.matmul(q_mat[:, :, x::self.n, :], torch.transpose(k_mat[:, :, x::self.n, :], 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
                inter = torch.matmul(inter, v_mat[:, :, x::self.n, :])
                #print(inter.shape)
                inter = torch.sum(inter, 2, keepdim = True)
                inter = inter.repeat(1, 1, self.num_frames, 1)
                temporal = torch.cat((temporal, inter), 2)
            
            print(temporal.shape)
            temporal_input = self.multimad_temp(rearrange(temporal, 'b h nf d -> b nf (h d)', h = self.num_heads))
            print(temporal_input.shape)

            q_mat = rearrange(self.q(temporal_input), 'b nf (h d) -> b h nf d', h = self.num_heads)
            v_mat = rearrange(self.k(temporal_input), 'b nf (h d) -> b h nf d', h = self.num_heads)
            k_mat = rearrange(self.v(temporal_input), 'b nf (h d) -> b h nf d', h = self.num_heads)

            temporal = self.softmax(torch.matmul(q_mat[:, :, 0::self.n, :], torch.transpose(k_mat[:, :, 0::self.n, :], 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
            temporal = torch.matmul(temporal, v_mat[:, :, 0::self.n, :])
            temporal = torch.sum(temporal, 2, keepdim = True)
           # print(temporal.shape)
          #  temporal = temporal.repeat(1, 1, self.num_frames, 1)
           # print(temporal.shape)
            
            # spatial calculation
            for x in range(1, self.n):
                # get all of the patches at that frame
                #print(torch.transpose(q_mat, 2 , 3).shape)
                #inter = self.softmax(torch.matmul(torch.transpose(q_mat, 2 , 3), k_mat[:, :, x::self.n, :]) / (math.sqrt(self.Dh) * self.num_heads))
                inter = self.softmax(torch.matmul(q_mat[:, :, x::self.n, :], torch.transpose(k_mat[:, :, x::self.n, :], 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
                inter = torch.matmul(inter, v_mat[:, :, x::self.n, :])
                #print(inter.shape)
                inter = torch.sum(inter, 2, keepdim = True)
            #    inter = inter.repeat(1, 1, self.num_frames, 1)
                temporal = torch.cat((temporal, inter), 2)

            print(temporal.shape)
            output = self.multi_mad_final(rearrange(temporal, 'b h nf d -> b nf (h d)', h = self.num_heads))
            print(output.shape)
            print(output)
            return output.repeat(1, self.num_frames, 1)
          
        


video = torch.randn(1, 5, 3, 224, 224)
p = 16
video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = p, p2 = p)
patch_embed = nn.Linear(768, 16)
video_to_feed = patch_embed(video)
n = 224 * 224 / (16 * 16)
print(n)
print(video_to_feed.shape)
new = dividedSpaceTimeAttention(8, 16, n, 5)
print(new(video_to_feed))