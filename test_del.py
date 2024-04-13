"""
class CustomLlamaEncoderLayer(nn.Module):
    ...:     def __init__(self):
    ...:         super().__init__()
    ...:         ##self.config = config
    ...:         self.input_layer = nn.Linear(256, 256)
    ...:         self.up_proj = nn.Linear(256, 256*2)
    ...:         self.down_proj = nn.Linear(256*2, 256)
    ...:         self.down_proj_2 = nn.Linear(256, 256//2)
    ...:         # norm
    ...:         self.norm = LlamaRMSNorm(256//2)
    ...:         self.act = nn.ReLU()
    ...:
    ...:     def forward(self, x,
    ...:         *args, **kwargs
    ...:         ):
    ...:
    ...:
    ...:         input_layer = self.input_layer(x)
    ...:         input_layer = self.act(input_layer)
    ...:
    ...:         up_proj = self.up_proj(x)
    ...:         up_proj = self.act(up_proj)
    ...:
    ...:         down_proj = self.down_proj(up_proj)
    ...:         down_proj = self.act(down_proj)
    ...:
    ...:         down_proj_2 = self.down_proj_2(down_proj)
    ...:         down_proj_2 = self.act(down_proj_2)
    ...:         down_proj_2 = self.norm(down_proj_2)
    ...:
    ...:
    ...:
    ...:         return down_proj_2
"""

import torch
from torch import nn

class LlamaRMSNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.alpha = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    

class CustomLlamaEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        ##self.config = config
        self.input_layer = nn.Linear(256, 256)
        self.up_proj = nn.Linear(256, 256*2)
        self.down_proj = nn.Linear(256*2, 256)
        self.down_proj_2 = nn.Linear(256, 256//2)
        # norm
        self.norm = LlamaRMSNorm(256//2)
        self.act = nn.ReLU()

    def forward(self, x,
        *args, **kwargs
        ):
        input_layer = self.input_layer(x)
        input_layer = self.act(input_layer)

        up_proj = self.up_proj(x)
        up_proj = self.act(up_proj)

        down_proj = self.down_proj(up_proj)
        down_proj = self.act(down_proj)

        down_proj_2 = self.down_proj_2(down_proj)
        down_proj_2 = self.act(down_proj_2)
        down_proj_2 = self.norm(down_proj_2)

        return down_proj_2
    

encoder_layer = CustomLlamaEncoderLayer()
dummy_data = torch.randn(1, 256)
output = encoder_layer(dummy_data)


# get name of the layer
print(encoder_layer.__class__.__name__)