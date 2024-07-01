from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from PIL import Image
import torch
import numpy as np
def build_transform(is_train):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=None,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation='bicubic',
        re_prob=0,
        re_mode="pixel", 
        re_count=1,
        mean=mean,
        std=std,
    )
    return transform

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

# 这里输入的期望是[3,h,w]
def saveImage(result , name):
    # result = pred.detach()
    # result = result.cpu()
    # result = result.numpy()
    # result = result[0]
    result = result.transpose(1,2,0)
    result = ((result * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN) * 255).astype(np.uint8)
    predImage = Image.fromarray(result)
    predImage.save(name)

def unshuffle( x, ids_restore):
    # x应该是什么样的？[B，Lm,C]?
    # ids呢？[B , L]
    # 我才masktokens现在变成了b , L-Lm , dim
    mask_token = torch.zeros(1, 1, 3)
    mask_tokens = mask_token.repeat(
        x.shape[0], ids_restore.shape[1] - x.shape[1], 1
    )
    # 然后现在x_变成了L
    x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
    )  # unshuffle

    return x_
def examine():
    input = "/home/zengshimao/code/Super-Resolution-Neural-Operator/data/test/ILSVRC2012_test_00000105.JPEG"
    img = Image.open(input).convert('RGB')
    transform = build_transform(True)
    img = transform(img)

    saveImage(img.numpy(),"/home/zengshimao/code/Super-Resolution-Neural-Operator/result/examine/gt.png")
    # c h w
    img_width = img.shape[2]
    img_height = img.shape[1]
    #  b c h w
    img = img.unsqueeze(0)
    # b c l
    img = img.reshape(1,3,-1)
    # b l c
    img = img.permute(0,2,1)

    img, mask, id_restore = random_masking(img, 0.75)
    
    aftermask = unshuffle(img , id_restore)
    aftermask = aftermask.permute(0,2,1)
    aftermask = aftermask.reshape(1,3,img_height,img_width)

    saveImage(aftermask[0].numpy(),"/home/zengshimao/code/Super-Resolution-Neural-Operator/result/examine/aftermask.png")

if __name__ =="__main__":
    examine()

    

