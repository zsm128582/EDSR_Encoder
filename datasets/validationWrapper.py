from datasets import register
import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image
import PIL

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def read_validation_labels(file_path):
    with open(file_path, 'r') as file:
        labels = file.readlines()
    # 去除每行末尾的换行符，并将标签转换为整数
    labels = [int(label.strip()) for label in labels]
    return np.array(labels)


def extract_sequence_number(file_path):
    # 拆分路径，获取文件名
    file_name = file_path.split('/')[-1]
    # 去掉前缀和后缀，提取中间的序号
    sequence_number = file_name.split('_')[-1].split('.')[0]
    return int(sequence_number)


if __name__ == "__main__":
    lables = read_validation_labels("/home/zengshimao/code/Super-Resolution-Neural-Operator/test/ILSVRC2012_validation_ground_truth.txt")
    print(lables[:10])

@register('validation_wrapper')
class ValidationWrapper(Dataset):
    def __init__(self, dataset , lables , augmentConfig=None, point_num = 300 , augment = False , istrain = True):
        self.dataset = dataset
        self.point_num = point_num
        self.augment = augment
        self.augmentConfigs = augmentConfig
        self.transform = self.build_transform(istrain)
        self.validationLables = lables
        
        self.validationLables = self.validationLables - 1 


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
        # img_width = img.shape[2]
        # img_height = img.shape[1]
        sequence_number = extract_sequence_number(img_path)

        # randomCoords = generate_random_points(img_width , img_height , self.point_num)
        # randomPoints = select_points_from_image(img , randomCoords)
        # torch.tensor(randomPoints)

        return {
            'img': img,
            'gt' : self.validationLables[sequence_number-1]
        }

    def build_transform(self ,is_train):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        # train transform
        if False:
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
 
        

 