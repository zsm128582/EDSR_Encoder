import math

import torch 
from torch import nn

from models import register
# class NestedTensor(object):
#     def __init__(self, tensors, mask):
#         self.tensors = tensors
#         self.mask = mask
#         if mask == 'auto':
#             self.mask = torch.zeros_like(tensors).to(tensors.device)
#             if self.mask.dim() == 3:
#                 self.mask = self.mask.sum(0).to(bool)
#             elif self.mask.dim() == 4:
#                 self.mask = self.mask.sum(1).to(bool)
#             else:
#                 raise ValueError("tensors dim must be 3 or 4 but {}({})".format(self.tensors.dim(), self.tensors.shape))
 
#     def imgsize(self):
#         res = []
#         for i in range(self.tensors.shape[0]):
#             mask = self.mask[i]
#             maxH = (~mask).sum(0).max()
#             maxW = (~mask).sum(1).max()
#             res.append(torch.Tensor([maxH, maxW]))
#         return res
 
#     def to(self, device):
#         # type: (Device) -> NestedTensor # noqa
#         cast_tensor = self.tensors.to(device)
#         mask = self.mask
#         if mask is not None:
#             assert mask is not None
#             cast_mask = mask.to(device)
#         else:
#             cast_mask = None
#         return NestedTensor(cast_tensor, cast_mask)
 
#     def to_img_list_single(self, tensor, mask):
#         assert tensor.dim() == 3, "dim of tensor should be 3 but {}".format(tensor.dim())
#         maxH = (~mask).sum(0).max()
#         maxW = (~mask).sum(1).max()
#         img = tensor[:, :maxH, :maxW]
#         return img
 
#     def to_img_list(self):
#         """remove the padding and convert to img list
#         Returns:
#             [type]: [description]
#         """
#         if self.tensors.dim() == 3:
#             return self.to_img_list_single(self.tensors, self.mask)
#         else:
#             res = []
#             for i in range(self.tensors.shape[0]):
#                 tensor_i = self.tensors[i]
#                 mask_i = self.mask[i]
#                 res.append(self.to_img_list_single(tensor_i, mask_i))
#             return res
 
#     @property
#     def device(self):
#         return self.tensors.device
 
#     def decompose(self):
#         return self.tensors, self.mask
 
#     def __repr__(self):
#         return str(self.tensors)
 
#     @property
#     def shape(self):
#         return {
#             'tensors.shape': self.tensors.shape,
#             'mask.shape': self.mask.shape
#         }
 

@register('positionalEmbedding')
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)

        dim_t = self.temperature ** (2 * (torch.div(dim_t,2,rounding_mode='trunc')) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
