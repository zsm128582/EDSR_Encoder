from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def random_masking(x, mask_ratio):

    h, w, c = x.shape
    noise = torch.rand(h, w, device=x.device)
    mask = (noise >= mask_ratio).float()
    mask = mask.unsqueeze(-1).repeat(1, 1, c)
    masked_img = img * mask
    return masked_img, mask


img_path = "/home/zengshimao/code/Super-Resolution-Neural-Operator/data/task3/n02085620_275.JPEG"
img = Image.open(img_path).convert("RGB")
img = torch.tensor(np.array(img)) / 255

masked_img, mask = random_masking(img, 0.90)
h, w, c = img.shape
plt.imsave("masked_img1.png", masked_img.numpy())

# image_array = Image.fromarray(masked_img.numpy().transpose(1,2,0))
# image_array.save("test.png")

# bl means batch like
# bl_img : b , h , w ,c
#  bl mask :b , h , w , c
bl_img = masked_img.unsqueeze(0)
bl_mask = mask.unsqueeze(0)
bl_img = bl_img.permute(0,3,1,2)

N ,C,H,W = bl_img.shape
grid = torch.zeros(N,h,w,2 , device= bl_img.device)
x = torch.linspace(-1, 1, W, device=bl_img.device)
y = torch.linspace(-1, 1, H, device=bl_img.device)
x_grid, y_grid = torch.meshgrid(x, y)
grid[:, :, :, 0] = x_grid.t()
grid[:, :, :, 1] = y_grid.t()

interpolated_pixels = F.grid_sample(bl_img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

final_imgs = bl_img.permute(0,2,3,1) + interpolated_pixels.permute(0,2,3,1) *(1- mask)

plt.imsave("grid_sample.png",final_imgs[0].numpy())


# #grid sample 插值方法

# bl_img = bl_img.permute(0,3,1,2)
# mask 

# # 普通线性插值的方法

# low_res_known_pixels = F.interpolate(
#     bl_img.permute(0, 3, 1, 2), scale_factor=0.3, mode="bilinear"
# )
# interpolated_pixels = F.interpolate(low_res_known_pixels, size=(h, w), mode="bilinear")

# plt.imsave("fullintepolate.png",interpolated_pixels.permute(0,2,3,1)[0].numpy())
# # mask为1代表保留
# # inte: b , c , h w
# final_imgs = bl_img + interpolated_pixels.permute(0, 2, 3, 1) * (1 - mask)

# res = final_imgs[0].numpy()
# plt.imsave("interpolate_img.png", res)
