import torch
import torch.nn as nn

from galerkin_transformer.model import SimpleAttention
from models.positionalEmbedding import PositionEmbeddingSine
from models import register

@register('selfAttentnio')
class SelfAttention(nn.Module):
    def __init__(self, n_feats=64):
        super(SelfAttention, self).__init__()
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
        self.n_feats = n_feats

        _norm_types = ["instance", "layer"]
        norm_type = _norm_types[1]
        attn_norm = True
        n_head = 8
        dropout = 0.1
        self.sa = SimpleAttention(
            n_head=n_head,
            d_model=n_feats,
            attention_type=_attention_types[1],
            pos_dim=-1,
            norm=attn_norm,
            norm_type=norm_type,
            dropout=0.0,
        )

        self.posEmbedding = PositionEmbeddingSine(
            num_pos_feats=n_feats // 2, normalize=True
        )
        self.dropout = nn.Dropout(dropout)

        self.layerNorm = nn.LayerNorm(n_feats)

    # x is supposed to be in shape of [ b , c , h  , w]
    def forward(self, x):
        b, c, h, w = x.shape

        assert c == self.n_feats , "输入数据的维度与预期不一致！"
        # at first ,transpose x to [b , h*w , c]
        pos = self.posEmbedding(x, None).permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        # attention
        x, _ = self.sa(query=x + pos, key=x + pos, value=x)
        x = x + self.dropout(x)
        x = self.layerNorm(x)

        # transpose x back to [ b , c , h  , w]
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, -1)
        return x


# @register("edsr")
# def make_edsr(
#     n_resblocks=32,
#     n_feats=256,
#     res_scale=0.1,
#     scale=2,
#     no_upsampling=False,
#     rgb_range=1,
# ):
#     args = Namespace()
#     args.n_resblocks = n_resblocks
#     args.n_feats = n_feats
#     args.res_scale = res_scale

#     args.scale = [scale]
#     args.no_upsampling = no_upsampling

#     args.rgb_range = rgb_range
#     args.n_colors = 3
#     return EDSR(args)