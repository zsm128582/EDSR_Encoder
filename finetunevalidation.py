import torch
from tqdm import tqdm
from timm.utils import accuracy
import torch.nn as nn
from models.classifitor import Classifier
from datasets.validationWrapper import ValidationWrapper
from datasets.image_folder import ImageFolder
from torch.utils.data import DataLoader 


def eval_finetune(loader , model):
    model.eval()
    pbar = tqdm(loader, leave=False, desc='val') 
    loss_Fn = nn.CrossEntropyLoss()  
    acc1_total = 0.
    acc5_total = 0.
    for batch in pbar :
        for k,v in batch.items():
            batch[k] = v.cuda(non_blocking = True)
        with torch.no_grad():
            pred  = model(batch["img"])
        # index =  torch.argmax(pred, 1, keepdim=False)
        # print("pred :" , index)
        # print("gt :",batch['gt'])
        acc1,acc5 = accuracy(pred,batch['gt'],topk=(1,5))
        print(acc1 , acc5)
        acc1_total += ( acc1 * batch["img"].shape[0] / 100)
        acc5_total += ( acc5 * batch["img"].shape[0] / 100)

        if False:
            pbar.set_description('val {:.4f}'.format(val_res.item()))
    acc1_total /= len(loader.dataset)
    acc5_total /= len(loader.dataset)
    print(acc1_total , acc5_total)
    return acc1_total , acc5_total


if __name__ == '__main__':
    encoder_spec = {}
    encoder_spec["name"] = "edsr-baseline"

    encoder_spec["args"] = {
        "no_upsampling" : True
    }
    # # model = Classifitor(encoder_spec)
    # model = myEncoder(encoder_spec)
    # print(model)
# augmentConfigs : 
#   input_size : 256
#   color_jitter : None
#   auto_augment : rand-m9-mstd0.5-inc1
#   # RE 的意思是 random eraser
#   reprob : 0
#   remode : pixel
#   recount : 1

    augmentConfigs = {}
    augmentConfigs['input_size'] = 256

    checkpoint_path = '/home/zengshimao/code/Super-Resolution-Neural-Operator/result/finetune-epoch30loss4.47.pth'
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    model = Classifier(encoder_spec,width=256)
    msg = model.load_state_dict(checkpoint['model']['sd'])
    print(msg)
    model = model.cuda()
    dataset = ValidationWrapper(ImageFolder("/home/zengshimao/code/Super-Resolution-Neural-Operator/data/validation",first_k=10000),augmentConfig=augmentConfigs,augment=True)
    dataloader = DataLoader(dataset, batch_size=8,
        shuffle=True, num_workers=8, pin_memory=True,persistent_workers=True)
    # paths = []

    # for batch in dataloader :
    #     gts = batch['gt']
    #     img_paths = batch['img_path']
    #     for gt , img_path in zip(gts , img_paths):
    #         if gt == 10:
    #             paths .append(img_path)
        
        # for img , gt , img_path in batch:
        #     if gt == 10 :
        #         img_paths.append(img_path)
    # print(paths)
    eval_finetune(dataloader , model)
    
