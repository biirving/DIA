# an implementation of the timeSformer model

from torch import nn, tensor
import torch
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
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
            q_mat = rearrange(self.q(input[:, 1:, :]), 'b nf (h d) -> b h nf d', h = self.num_heads)
            v_mat = rearrange(self.k(input[:, 1:, :]), 'b nf (h d) -> b h nf d', h = self.num_heads)
            k_mat = rearrange(self.v(input[:, 1:, :]), 'b nf (h d) -> b h nf d', h = self.num_heads)


            # the class token has to attend to the entire input (don't need multiple heads)
            # So the q matrix will be the query of the first row (the class token) which will
            # 'attend' to all of the key values
            q_mat_cls_tkn = rearrange(self.q(input[:, 0:1, :]), 'b nf (h d) -> b h nf d', h = self.num_heads)
            v_mat_cls_tkn = rearrange(self.v(input), 'b nf (h d) -> b h nf d', h = self.num_heads)
            k_mat_cls_tkn = rearrange(self.k(input), 'b nf (h d) -> b h nf d', h = self.num_heads)
            inter_cls_tkn = self.softmax(torch.matmul(q_mat_cls_tkn, torch.transpose(k_mat_cls_tkn, 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
            inter_cls_tkn = torch.matmul(inter_cls_tkn, v_mat_cls_tkn)
 
            # First, we calculate temporal attention, multiplying the q vector being processed by every k vector at that 
            # frame in subsequent time steps

            # at this point, the q matrix contains all of the query vectors to be processed
            # but each row will be multiplied by a different set of the k vectors, because they are being compared to the other
            # keys at that timeframe
            temporal = self.softmax(torch.matmul(q_mat[:, :, 1::self.n, :], torch.transpose(k_mat[:, :, 1::self.n, :], 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
            temporal = torch.matmul(temporal, v_mat[:, :, 1::self.n, :])
            temporal = torch.sum(temporal, 2, keepdim = True)
            
            # temporal calculation
            for x in range(2, self.n):
                # get all of the patches at that frame
                inter = self.softmax(torch.matmul(q_mat[:, :, x::self.n, :], torch.transpose(k_mat[:, :, x::self.n, :], 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
                inter = torch.matmul(inter, v_mat[:, :, x::self.n, :])
                inter = torch.sum(inter, 2, keepdim = True)
                #inter = inter.repeat(1, 1, self.num_frames, 1)
                temporal = torch.cat((temporal, inter), 2)
            
            temporal = temporal.repeat(1, 1, self.num_frames, 1)
            temporal_input = self.multimad_temp(rearrange(temporal, 'b h nf d -> b nf (h d)', h = self.num_heads))

            q_mat = rearrange(self.q(temporal_input), 'b nf (h d) -> b h nf d', h = self.num_heads)
            v_mat = rearrange(self.k(temporal_input), 'b nf (h d) -> b h nf d', h = self.num_heads)
            k_mat = rearrange(self.v(temporal_input), 'b nf (h d) -> b h nf d', h = self.num_heads)

            temporal = self.softmax(torch.matmul(q_mat[:, :, 0:self.n, :], torch.transpose(k_mat[:, :, 0:self.n, :], 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
            temporal = torch.matmul(temporal, v_mat[:, :, 0:self.n, :])
            temporal = torch.sum(temporal, 2, keepdim = True)


            # spatial calculation
            for x in range(1, self.num_frames):
                inter = self.softmax(torch.matmul(q_mat[:, :, x:x+self.n, :], torch.transpose(k_mat[:, :, x::self.n, :], 2, 3)) / (math.sqrt(self.Dh) * self.num_heads))
                inter = torch.matmul(inter, v_mat[:, :, x::self.n, :])
                inter = torch.sum(inter, 2, keepdim = True)
                temporal = torch.cat((temporal, inter), 2)

            temporal = temporal.repeat(1, 1, self.n, 1)
            temporal = torch.cat((inter_cls_tkn, temporal), 2)
            output = self.multi_mad_final(rearrange(temporal, 'b h nf d -> b nf (h d)', h = self.num_heads))
            return output
            
        

class EncoderBlock(nn.Module):
    def __init__(self, num_heads, dim, n, num_frames):
        super(EncoderBlock, self).__init__()
        # number of attention heads
        self.num_heads = num_heads
        # number of attention blocks
        self.dim = int(dim)
        self.num_frames = num_frames
        # layer normalization, to stabilize the gradients
        # should not depend on batch size
        self.n = int(n)
        #self.norm = nn.LayerNorm(normalized_shape = (self.n, self.dim))
        self.norm = nn.LayerNorm(normalized_shape = (self.num_frames * self.n + 1, dim), elementwise_affine = True)
        self.attention = dividedSpaceTimeAttention(self.num_heads, dim, self.n, num_frames)
        self.dh = int(self.dim / self.num_heads)
        self.mlp = nn.Linear(dim, dim)

        
    def forward(self, input):
        #whoa = self.norm(input)
        uhOh = self.attention.forward(input)
        uhHuh = self.norm(uhOh + input)
        toAdd = uhOh + input    
        output = self.mlp(uhHuh)
        output += toAdd
        return output


# You should be able to feed an input of any batch size to the model
class timeSformer(nn.Module):

    def __init__(self, height, width, num_frames, patch_res, dim, num_classes, batch_size):
        super(timeSformer, self).__init__()
        self.checkPass = True
        self.height = height
        self.width = width
        self.channels = 3
        self.num_classes = num_classes
        self.patch_res = patch_res 
        self.dim = int(dim)
        self.patch_dim = self.channels * patch_res * patch_res
        self.n = int((height * width) / (patch_res ** 2))


        self.patchEmbed = nn.Sequential(
            Rearrange('b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = patch_res, p2 = patch_res),
            nn.Linear(self.patch_dim, dim),)
        # the class token serves as a representation of the entire sequence
        # should attend to the entire sequence (so, it actually doesn't need to be stored with 
        # the other vectors? as long as it is weighted sum?)
        self.classtkn = nn.Parameter(torch.randn(batch_size, 1, dim))
        # this will be concated to the end of the input before positional embedding
        # should the class token be randomonly initialized? or the same across batches? See what happens during training

        # the positional embedding should be applied based on what 
        self.pos_embed = nn.Parameter(torch.randn(batch_size, num_frames * (self.n) + 1, dim))
        self.encoderBlocks = nn.ModuleList([EncoderBlock(num_heads = 8, dim = dim, n = self.n, num_frames=num_frames) for i in range(8)])

        
        self.mlpHead = nn.Sequential(nn.LayerNorm(dim), nn.GELU(), nn.Linear(self.dim, num_classes))
        self.dropout = nn.Dropout(0.1)


            

    def forward(self, vid):
        input = self.patchEmbed(vid)
        input = torch.cat((self.classtkn, input), dim = 1)
        input += self.pos_embed
        input = self.dropout(input)
        for encoder in self.encoderBlocks:
            output = encoder.forward(input)
            input = output
        out = self.mlpHead(output[:, 0])
        return out



