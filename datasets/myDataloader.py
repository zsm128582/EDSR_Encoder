from datasets import register
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import generate_random_points
from utils import select_points_from_image

from torchvision import transforms
from PIL import Image

@register('random-n-dataloader')
class myDataloader(Dataset):
    def __init__(self, dataset, point_num = 300 , augment = False):
        self.dataset = dataset
        self.point_num = point_num
        self.augment = augment
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        img = transforms.ToTensor()(
                Image.open(img_path).convert('RGB'))
        # 做一些图像增强咯

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
 

 