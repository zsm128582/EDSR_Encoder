from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def unshuffle(x, ids_restore):
    # x应该是什么样的？[B，Lm,C]?
    # ids呢？[B , L]
    # 我才masktokens现在变成了b , L-Lm , dim
    mask_token = nn.Parameter(torch.zeros(1, 1, 3))
    mask_tokens = mask_token.repeat(
        x.shape[0], ids_restore.shape[1] - x.shape[1], 1
    )
    # 然后现在x_变成了L
    x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
    )  # unshuffle

    return x_

def random_masking( x, mask_ratio):
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

img_path = "/home/zengshimao/code/Super-Resolution-Neural-Operator/data/task3/n02085620_275.JPEG"
img = Image.open(img_path).convert('RGB')
img = torch.tensor(np.array(img))
img = img.reshape(1,-1,3)
x, mask, id_restore = random_masking(img.unsqueeze(0), 0.75)


restoreImage = unshuffle(x,id_restore)


restoreImage = restoreImage.view(1, 224,224, 3)

image_array = Image.fromarray(restoreImage[0])
image_array.save("test.png")





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