import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

import models

from models import register
from typing import Optional
from models.selfAttention import SelfAttention
from models.ffn_layer import FFNLayer  
from models.pos_embed import get_2d_sincos_pos_embed
from functools import partial

# from positionalEmbedding import NestedTensor


@register("random_N_Classifitor")
class Classifier(nn.Module):
    def __init__(self, encoder_spec, width=256, blocks=16 , num_classes = 1000) -> None:
        super().__init__()
        hidden_dim = 64
        self.width = width
        self.encoder = models.make(encoder_spec)
        self.sa1 = SelfAttention(n_feats=hidden_dim)
        self.sa2 = SelfAttention(n_feats=hidden_dim)

        self.cls_token = nn.Parameter(torch.zeros(1,1,hidden_dim))
        torch.nn.init.normal_(self.cls_token, std=.02)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.width * width + 1, hidden_dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.width, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.mlp = MLP(
            input_dim=hidden_dim, hidden_dim=hidden_dim * 2, output_dim=3, num_layers=3
        )

        self.norm = nn.LayerNorm(hidden_dim,eps=1e-6)

        self.head = nn.Linear(hidden_dim, num_classes)

    def forwardEncoder(self, img):
        x = self.encoder(img)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1, x.size(3))
        x = x + self.pos_embed[:,1:,:]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.sa1(x)
        x = self.sa2(x)
        return x


    def forward(self, img):
        224*224,64
        x= self.forwardEncoder(img)

        x = self.norm(x)
        cls_token = x[:,0]
        # 64 - > 1000 
        pred = self.head(cls_token)
        return pred


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

