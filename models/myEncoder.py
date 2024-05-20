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
from models.selfAttention import SelfAttention
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


        self.decoder = myDecoder(attentionBlockNum= 4 ,embed_dim=hidden_dim)


    def gen_feat(self, img):
        self.img = img
        self.feat = self.encoder(img)
        return self.feat

    def forward(self, img, coords):
        self.coords = coords
        #  1. 用edsr提取图像特征
        # timer_start = torch.cuda.Event(enable_timing=True)
        # timer_end = torch.cuda.Event(enable_timing=True)
        # timer_start.record()

        self.gen_feat(img)

        # timer_end.record()
        # torch.cuda.synchronize()
        # getFeat_time = timer_start.elapsed_time(timer_end)

        # 2. sampling        or    mask (和 mae 有什么区别？)

        # timer_start.record()

        samples = self.get_samples(self.feat, self.coords)

        # samples = NestedTensor(tensors=samples ,  mask=coords)

        # 输入 ： x : [ b , c , h , w]  , mask : [ b , h , w]
        pos = self.posEmbedding(self.feat, None)

        # 然后对pos进行sampling
        samples_pos = self.get_samples(pos, self.coords)

        # timer_end.record()
        # torch.cuda.synchronize()
        # sampleTime = timer_start.elapsed_time(timer_end)



        # 3. garkerlin + pos embedding


        # query=samples + samples_pos
        # key=samples + samples_pos
        # value=samples 
        #  q , k , v
        # timer_start.record()

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

        restoreImage = self.decoder(img.shape , self.coords , samples , pos)
        


        #然后这里写pos embedding
        #然后相加

        
        #然后经过几层 transformer  ， mae用了8层？ 
        #然后就经过一个mlp

        # timer_end.record()
        # torch.cuda.synchronize()
        # attention_time = timer_start.elapsed_time(timer_end)

        # print(f" generate feature time : {getFeat_time:.4f}ms,  get sample time {sampleTime:.4f}ms, attention Time: {attention_time:.4f}ms")

        return restoreImage
        #  add residual layer
        # 4. decoder
        
        # a simple mlp ?

        # 5. loss

        # or ...

    # input : img: tensor[ b , c , w , h ]   coords : [b , n , 2 ]
    # output : tensor[ b , c , len(coords)]
    def get_samples(self ,img, coords):
        # getsample_start = torch.cuda.Event(enable_timing=True)
        # getsample_end = torch.cuda.Event(enable_timing=True)
        # getsample_start.record()

        batch_size , chennel , _ , _ = img.shape
        point_num = coords.shape[1]
        select_points = torch.empty((batch_size , point_num, chennel),device=img.device)
        for b in range(batch_size):
            select_points[b]= self.select_points_from_image(img[b],coords[b])

        # getsample_end.record()
        # torch.cuda.synchronize()
        # sampleTime = getsample_start.elapsed_time(getsample_end)
        # print(f"sample Time: {sampleTime:.4f}ms")

        return select_points
    

    #input : coords :  tensor[n , 2]
    # image : tensor [3, h , w ]
    def select_points_from_image(self,image, coordinates):
        image =  image.permute(1,2,0)
        coordinates = coordinates.long()
        # max_x = torch.max(coordinates[:, 0])
        # max_y = torch.max(coordinates[:, 1])
    
        # print(f"Max x coordinate: {max_x}, Max y coordinate: {max_y}")
        # assert torch.all((coordinates[:, 0] >= 0) & (coordinates[:, 0] < image.shape[0])), "x_coords out of bounds"
        # assert torch.all((coordinates[:, 1] >= 0) & (coordinates[:, 1] < image.shape[1])), "y_coords out of bounds"
        x_coords = coordinates[:, 0]
        y_coords = coordinates[:, 1]
        selected_points = image[x_coords, y_coords]

        # selected_points = torch.zeros((coordinates.shape[0] , image.shape[0]),device = image.device)
        # image = image.permute(1,2,0)
        # for index ,  coord in enumerate(coordinates):
        #     selected_points[index] = image[coord[0] , coord[1]] # Assuming image is a 2D list or array
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


class myDecoder(nn.Module):
    def __init__ (self , attentionBlockNum = 8 , embed_dim = 64):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([
            SelfAttention(n_feats = embed_dim) for i in range(attentionBlockNum)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = MLP(embed_dim , embed_dim*2 , 3 , 3 )


    def forward(self , image_shape , coordinates , processed_points , pos):
        
        restoreImage = self.restore_points_to_image(image_shape, coordinates , processed_points)
        restoreImage  = restoreImage + pos
        for block  in self.decoder_blocks:
            restoreImage = block(restoreImage)
            # 这里要写ffn ， norm？
        restoreImage = restoreImage.permute(0,2,3,1)
        restoreImage = self.decoder_norm(restoreImage)
        restoreImage = self.decoder_pred(restoreImage)
        restoreImage = restoreImage.permute(0 ,3 ,1 ,2 )
        return restoreImage


    # def unsuffledImage(self,image_shape , coordinates , processed_points):
    #     _, H, W = image_shape
    #     C = processed_points.shape[1]  # embed_dim
    #     restored_image = torch.zeros((C, H, W), device=processed_points.device)
    #     coordinates = coordinates.long()

    #     x_coords = coordinates[:, 0]
    #     y_coords = coordinates[:, 1]

    #     restored_image[:, x_coords, y_coords] = processed_points.t()

    #     return restored_image

    def restore_points_to_image(self ,batch_image_shape, batch_coordinates, batch_processed_points):
        """
        将批处理的处理后的点放回它们原来的位置
        
        参数：
        - batch_image_shape (tuple): 批处理图像的形状，格式为 (B, C, H, W)
        - batch_coordinates (Tensor): 批处理坐标张量，形状为 [B, n, 2]
        - batch_processed_points (Tensor): 批处理处理后的点，形状为 [B, n, embed_dim]
        
        返回：
        - batch_restored_image (Tensor): 恢复后的批处理图像，形状为 (B, embed_dim, H, W)
        """
        B, _, H, W = batch_image_shape
        embed_dim = batch_processed_points.shape[2]
        
        # 初始化一个空的图像，形状为 (B, embed_dim, H, W)
        batch_restored_image = torch.zeros((B, embed_dim, H, W), device=batch_processed_points.device)
        
        # 转换坐标为long类型
        batch_coordinates = batch_coordinates.long()
        
        for b in range(B):
            coordinates = batch_coordinates[b]
            processed_points = batch_processed_points[b]
            
            x_coords = coordinates[:, 0]
            y_coords = coordinates[:, 1]
            
            batch_restored_image[b, :, x_coords, y_coords] = processed_points.t()
        
        return batch_restored_image
    