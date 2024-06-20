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
from models.ffn_layer import FFNLayer
from models.pos_embed import get_2d_sincos_pos_embed
from timm.models.vision_transformer import PatchEmbed, Block

# from positionalEmbedding import NestedTensor


@register("random_N_encoder")
class myEncoder(nn.Module):
    def __init__(self, encoder_spec, width=256, blocks=16) -> None:
        super().__init__()
        hidden_dim = 64
        self.encoder = models.make(encoder_spec)
        self.sa1 = SelfAttention(n_feats=hidden_dim)
        self.sa2 = SelfAttention(n_feats=hidden_dim)

        # TODO:这里的224不要写死
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 224 * 224, hidden_dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 224, cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.mlp = MLP(
            input_dim=hidden_dim, hidden_dim=hidden_dim * 2, output_dim=3, num_layers=3
        )

        self.decoder = myDecoder(attentionBlockNum=4, embed_dim=hidden_dim)

    def gen_feat(self, img):
        return self.encoder(img)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # argsort 返回的是索引值
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forwardEncoder(self, img):

        # [b,c,h,w]
        x = self.gen_feat(img)

        # TODO:在这里将x变成[b,hw,c]
        # x = torch.einsum("bchw->blc", x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1, x.size(3))

        x = x + self.pos_embed

        x, mask, id_restore = self.random_masking(x, 0.75)

        x = self.sa1(x)
        x = self.sa2(x)

        return x, mask, id_restore

    def forwardDecoder(self, img, id_restore, mask):

        return self.decoder(img, id_restore, mask)

    def forward(self, img):
        h = img.shape[2]
        w = img.shape[3]
        x, mask, id_restore = self.forwardEncoder(img)
        pred = self.forwardDecoder(x, id_restore, mask)
        # pred 是 【b,l,3】
        # 在这里变成[b,3,h,w]

        return pred, mask

    # input : img: tensor[ b , c , w , h ]   coords : [b , n , 2 ]
    # output : tensor[ b , c , len(coords)]
    def get_samples(self, img, coords):
        # getsample_start = torch.cuda.Event(enable_timing=True)
        # getsample_end = torch.cuda.Event(enable_timing=True)
        # getsample_start.record()

        batch_size, chennel, _, _ = img.shape
        point_num = coords.shape[1]
        select_points = torch.empty((batch_size, point_num, chennel), device=img.device)
        for b in range(batch_size):
            select_points[b] = self.select_points_from_image(img[b], coords[b])

        # getsample_end.record()
        # torch.cuda.synchronize()
        # sampleTime = getsample_start.elapsed_time(getsample_end)
        # print(f"sample Time: {sampleTime:.4f}ms")

        return select_points

    # input : coords :  tensor[n , 2]
    # image : tensor [3, h , w ]
    def select_points_from_image(self, image, coordinates):
        image = image.permute(1, 2, 0)
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


class myDecoder(nn.Module):
    def __init__(self, attentionBlockNum=8, embed_dim=64):
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [SelfAttention(n_feats=embed_dim) for i in range(attentionBlockNum)]
        )
        # TODO:这里也一样！
        self.h = 224
        self.w = 224
        self.intepolate = False
        self.patchAttention = True

        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = MLP(embed_dim, embed_dim * 2, 3, 3)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # TODO:这里的224不要写死
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 224 * 224, embed_dim), requires_grad=False
        )
        temp = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 224, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(temp).float().unsqueeze(0))

        if self.patchAttention:
            self.patch_size = 16
            self.patchEmbed = PatchEmbed(
                self.h, patch_size=self.patch_size, in_chans=embed_dim, embed_dim=1024
            )
            self.num_patches = self.patchEmbed.num_patches
            self.patch_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, 1024), requires_grad=False
            )  # fixed sin-cos embedding
            temp = get_2d_sincos_pos_embed(
                self.patch_pos_embed.shape[-1],
                int(self.num_patches**0.5),
                cls_token=False,
            )
            self.patch_pos_embed.data.copy_(torch.from_numpy(temp).float().unsqueeze(0))
            self.blocks = nn.ModuleList(
                [
                    Block(
                        1024,
                        num_heads=16,
                        mlp_ratio=4.0,
                        qkv_bias=True,
                        norm_layer=nn.LayerNorm,
                    )
                    for i in range(2)
                ]
            )
            self.patchNorm = nn.LayerNorm(1024)
            self.patch_pred = nn.Linear(1024 , self.patch_size ** 2 * 3, bias=True)

    # x : b l c
    # return : b 3 h w
    def forward(self, x, ids_restore, mask):
        # mask : [b , l]  1 denote mask , 0 denote left
        # 现在变成了[b,l,c]
        restoreImage = self.unshuffle(x, ids_restore)
        b, l, c = restoreImage.shape
        # restore 之后依然是blc
        if self.intepolate:
            restoreImage = restoreImage.view(b, self.h, self.w, c)
            mask = mask.view(b, self.h, self.w)

            low_res_known_pixels = F.interpolate(
                restoreImage.permute(0, 3, 1, 2), scale_factor=0.5, mode="bilinear"
            )
            low_res_mask = F.interpolate(
                (1 - mask).unsqueeze(1).float(), scale_factor=0.5, mode="bilinear"
            )

            interpolated_pixels = F.interpolate(
                low_res_known_pixels, size=(self.h, self.w), mode="bilinear"
            )
            interpolated_mask = F.interpolate(
                low_res_mask, size=(self.h, self.w), mode="bilinear"
            )

            interpolated_mask = interpolated_mask.expand(-1, c, -1, -1)
            # 这里的mask使用插值后的mask？还是插值后的mask？

            final_imgs = (
                restoreImage.permute(0, 3, 1, 2)
                + interpolated_pixels * interpolated_mask
            )
            # final_imgs = restoreImage + interpolated_pixels * (mask.unsqueeze(-1) > 0).float()

            restoreImage = final_imgs.permute(0, 2, 3, 1).view(b, l, c)
            mask = mask.view(b, l)

        if self.patchAttention:
            # 首先先做一层全局的galerkin？这里先不写，不然的话还得加上一层pos embeding

            # 然后把他们分块
            # 在这里，patch有两种方法，第一种就是直接使用einsum的方法通过重排，16*16*64 -》 16384 16*16*3 -》 768
            # 重排后的长度还是太大了，还需要通过一个ffn降维到大约2048或1024

            # 第二种就是通过2d卷积的方式进行 , restoreImage : BLC ， 而x需要为 b ,c ,h ,w
            restoreImage = restoreImage.reshape(b, self.h, self.w, c)
            # 现在x变成bchw了
            x = restoreImage.permute(0, 3, 1, 2)

            # 经过patchify之后会变成blc , l = 14*14
            x = self.patchEmbed(x)
            # 然后加上分块后的pos embedding
            x = x + self.patch_pos_embed
            # 套用一下传统的attention
            for blk in self.blocks:
                x = blk(x)
            # 最后展开或者不展开都行
            # 最后仍然是b L C咯 - > 最好变成b l c ?
            # 首先需要把C - > 16*16*3 
            # 现在是： b , L , 16*16*3
            x = self.patchNorm(x)
            x = self.patch_pred(x)

            # 我这里需要考虑的是 ： 是把groud truth patchify 还是把 pred unpatchify
            # 现在x : b 3 h w
            x = self.unpatchify(x)
            return x
        else:
            # 这一块就是正常的使用galerkin
            restoreImage = restoreImage + self.pos_embed
            for block in self.decoder_blocks:
                restoreImage = block(restoreImage)

            restoreImage = self.decoder_norm(restoreImage)
            restoreImage = self.decoder_pred(restoreImage)
            restoreImage = restoreImage.reshape(restoreImage.size(0), self.h, self.w, restoreImage.size(2))
            restoreImage = restoreImage.permute(0, 3, 1, 2)
            return restoreImage

    def unshuffle(self, x, ids_restore):
        # x应该是什么样的？[B，Lm,C]?
        # ids呢？[B , L]
        # 我才masktokens现在变成了b , L-Lm , dim
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        # 然后现在x_变成了L
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle

        return x_
    

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def restore_points_to_image(
        self, batch_image_shape, batch_coordinates, batch_processed_points
    ):
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
        batch_restored_image = torch.zeros(
            (B, embed_dim, H, W), device=batch_processed_points.device
        )

        # 转换坐标为long类型
        batch_coordinates = batch_coordinates.long()

        for b in range(B):
            coordinates = batch_coordinates[b]
            processed_points = batch_processed_points[b]

            x_coords = coordinates[:, 0]
            y_coords = coordinates[:, 1]

            batch_restored_image[b, :, x_coords, y_coords] = processed_points.t()

        return batch_restored_image
