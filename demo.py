import argparse
import os
from PIL import Image
import PIL
import numpy as np
import torch
from torchvision import transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models

from utils import generate_random_points

import matplotlib.pyplot as plt



def build_transform(is_train):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # input_size : 224
    # color_jitter : None
    # auto_augment : rand-m9-mstd0.5-inc1
    # # RE 的意思是 random eraser
    # reprob : 0
    # remode : pixel
    # recount : 1
    # train transform

    # this should always dispatch to transforms_imagenet_train
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

    # # eval transform
    # # TODO: validatiopn的config还没写好！
    # t = []
    # if self.augmentConfigs["input_size"] <= 224:
    #     crop_pct = 224 / 256
    # else:
    #     crop_pct = 1.0
    # size = int(self.augmentConfigs["input_size"] / crop_pct)
    # t.append(
    #     transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    # )
    # t.append(transforms.CenterCrop(self.augmentConfigs["input_size"]))

    # t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(mean, std))
    # return transforms.Compose(t)

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

if __name__ == '__main__':
    model_path = "/home/zengshimao/code/Super-Resolution-Neural-Operator/save/_train-randomN/epoch-last.pth"
    # input = "/home/zengshimao/code/Super-Resolution-Neural-Operator/data/test/ILSVRC2012_test_00000245.JPEG"
    input = "/home/zengshimao/code/Super-Resolution-Neural-Operator/data/test/ILSVRC2012_test_00000023.JPEG"
    img = Image.open(input).convert('RGB')
    transform = build_transform(True)
    img = transform(img)

    img_width = img.shape[2]
    img_height = img.shape[1]
 
    saveImage(img.numpy(),"/home/zengshimao/code/Super-Resolution-Neural-Operator/result/patchAttention/gt.png")
    # result = img.numpy()
    # result = result.transpose(1,2,0)
    # result = (result * 255).astype(np.uint8)
    # predImage = Image.fromarray(result)
    # predImage.save()


    # randomCoords = generate_random_points(img_width , img_height , 8000)

    # randomPoints = select_points_from_image(img , randomCoords)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    

    model = models.make(torch.load(model_path)['model'], load_sd=True).cuda()
    model.eval()


    # make it batch like
    img = img.unsqueeze(0)
    # randomCoords = randomCoords.unsqueeze(0)

    img = img.cuda(non_blocking=True)
    # randomCoords = randomCoords.cuda(non_blocking=True)

    pred , mask = model(img)

    mask = mask.view(pred.shape[-2],pred.shape[-1])

    print(torch.sum(mask))

    aftermask = img * (1- mask)


    aftermask = aftermask.detach().cpu().numpy()

    saveImage(aftermask[0] , "/home/zengshimao/code/Super-Resolution-Neural-Operator/result/patchAttention/aftermask.png")

    result = pred.detach()
    result = result.cpu()
    result = result.numpy()
    result = result[0]

    saveImage(result , "/home/zengshimao/code/Super-Resolution-Neural-Operator/result/patchAttention/pred.png")







