import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

import models

from models import register
from typing import Optional
from galerkin_transformer.model import SimpleAttention
from models.positionalEmbedding import PositionEmbeddingSine
# from positionalEmbedding import NestedTensor


@register("random_N_encoder")
class myEncoder(nn.Module):
    def __init__(self, encoder_spec, width=256, blocks=16) -> None:
        super().__init__()
        self.encoder = models.make(encoder_spec)

        _attention_types = [
            "linear",
            "galerkin",
            "global",
            "causal",
            "fourier",
            "softmax",
            "integral",
            "local",
        ]
        attention_type = _attention_types[1]
        _norm_types = ["instance", "layer"]
        norm_type = _norm_types[1]
        attn_norm = True
        hidden_dim = 64
        n_head = 8
        dropout = 0.1
        dim_feedforward = hidden_dim * 2
        pre_norm = True




        self.sa1 = SimpleAttention(
            n_head=n_head,
            d_model=hidden_dim,
            attention_type=_attention_types[1],
            pos_dim=-1,
            norm=attn_norm,
            norm_type=norm_type,
            dropout=0.0,
        )

        self.sa2 = SimpleAttention(
            n_head=n_head,
            d_model=hidden_dim,
            attention_type=_attention_types[1],
            pos_dim=-1,
            norm=attn_norm,
            norm_type=norm_type,
            dropout=0.0,
        )

        self.posEmbedding = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2, normalize=True
        )

        self.dropout = nn.Dropout(dropout)

        self.layerNorm = nn.LayerNorm(hidden_dim)

        self.ffn = FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
        self.mlp = MLP(input_dim= hidden_dim,hidden_dim= hidden_dim , output_dim= 3 , num_layers= 3)
    def gen_feat(self, img):
        self.img = img
        self.feat = self.encoder(img)
        return self.feat

    def forward(self, img, coords):
        self.coords = coords
        #  1. 用edsr提取图像特征
        self.gen_feat(img)

        # 2. sampling        or    mask (和 mae 有什么区别？)

        samples = self.get_samples(self.feat, self.coords)

        # samples = NestedTensor(tensors=samples ,  mask=coords)

        # 输入 ： x : [ b , c , h , w]  , mask : [ b , h , w]
        pos = self.posEmbedding(self.feat, None)

        # 然后对pos进行sampling
        samples_pos = self.get_samples(pos, self.coords)
        # 3. garkerlin + pos embedding

        # query=samples + samples_pos
        # key=samples + samples_pos
        # value=samples 
        #  q , k , v

        output , _ = self.sa1(
            query=samples + samples_pos, key=samples + samples_pos, value=samples
        )
        
        samples = samples +  self.dropout(output)
        samples = self.layerNorm(samples)

        output , _ = self.sa2(
            query=samples + samples_pos, key=samples + samples_pos, value=samples
        )

        samples = samples +  self.dropout(output)
        samples = self.layerNorm(samples)

        samples = self.ffn(samples)
        
        pred = self.mlp(samples)
        return pred
        #  add residual layer
        # 4. decoder
        
        # a simple mlp ?

        # 5. loss

        # or ...

    # input : img: tensor[ b , c , w , h ]   coords : [b , n ,2 ]
    # output : tensor[ b , c , len(coords)]
    def get_samples(self ,img, coords):
        batch_size , chennel , _ , _ = img.shape
        point_num = coords.shape[1]
        select_points = torch.empty((batch_size , point_num, chennel),device=img.device)
        for b in range(batch_size):
            select_points[b]= self.select_points_from_image(img[b],coords[b])
        return select_points
    

        #input : coords :  tensor[n , 2]
    #  image : tensor [3, h , w ]
    #  output  : points :  tensor [ 3, n ] ?   / [ n ,3 ] 
    def select_points_from_image(self,image, coordinates):
        selected_points = torch.zeros((coordinates.shape[0] , image.shape[0]),device = image.device)
        image = image.permute(1,2,0)
        for index ,  coord in enumerate(coordinates):
            selected_points[index] = image[coord[0] , coord[1]] # Assuming image is a 2D list or array
        return selected_points
    


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
