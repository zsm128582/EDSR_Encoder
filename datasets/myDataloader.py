from datasets import register
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import generate_random_points
from utils import select_points_from_image

from torchvision import transforms
from PIL import Image
import PIL

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



@register('random-n-dataloader')
class myDataloader(Dataset):
    def __init__(self, dataset, point_num = 300 , augment = False , istrain = True , **configs):
        self.dataset = dataset
        self.point_num = point_num
        self.augment = augment
        self.augmentConfigs = configs["augmentConfigs"]
        self.transform = self.build_transform(istrain)


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        # img = transforms.ToTensor()(
        #         Image.open(img_path).convert('RGB'))
        img = Image.open(img_path).convert('RGB')
        # 做一些图像增强

        if self.augment : 
            img = self.transform(img)
        else :
            img = transforms.ToTensor(img)
        img_width = img.shape[2]
        img_height = img.shape[1]


        randomCoords = generate_random_points(img_width , img_height , self.point_num)
        randomPoints = select_points_from_image(img , randomCoords)
        # torch.tensor(randomPoints)

        return {
            'img': img,
            'coord': randomCoords,
            'gt': randomPoints
        }

    def build_transform(self ,is_train):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        # train transform
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=self.augmentConfigs["input_size"],
                is_training=True,
                color_jitter=self.augmentConfigs["color_jitter"],
                auto_augment=self.augmentConfigs["auto_augment"],
                interpolation='bicubic',
                re_prob=self.augmentConfigs["reprob"],
                re_mode=self.augmentConfigs["remode"],
                re_count=self.augmentConfigs["recount"],
                mean=mean,
                std=std,
            )
            return transform

        # eval transform
        # TODO: validatiopn的config还没写好！
        t = []
        if self.augmentConfigs["input_size"] <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(self.augmentConfigs["input_size"] / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(self.augmentConfigs["input_size"]))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)
 

 