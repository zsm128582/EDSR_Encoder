import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader , random_split
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.classifitor import Classifier
import sys

import math

import datasets
import models
import utils
from test import eval_finetune
from scheduler import GradualWarmupScheduler
from datasets.validationWrapper import ValidationWrapper , read_validation_labels
from datasets.image_folder import ImageFolder
from timm.models.layers import trunc_normal_



def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    # for k, v in dataset[0].items():
    #     log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True,persistent_workers=True)
    return loader


def make_data_loaders():
    # TODO:change a root path
    dataset = ValidationWrapper(ImageFolder(config.get('train_rootPath')),lables=read_validation_labels(config.get('lable_path')),augmentConfig=config['augmentConfigs'],augment=True)
    train_size = int(0.8 * len ( dataset))
    test_size = len(dataset) - train_size
    train_dataset , test_dataset = random_split(dataset , [train_size , test_size])
    train_loader = DataLoader(train_dataset, batch_size=config.get('train_batchsize'),
        shuffle=True, num_workers=8, pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(test_dataset , batch_size=config.get('val_batchsize'),shuffle=False,num_workers=8, pin_memory=True,persistent_workers=True)
    return train_loader, val_loader


def interpolate_pos_embed(model , checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]

        # 这里需要改成实际输入的点数
        pixelNum = model.width ** 2
        # extra就是cls token的个数，就是总的token树剪掉像素点的个数，剩下的就是cls token的数量
        num_extra_tokens = model.pos_embed.shape[-2] - pixelNum
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(pixelNum ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed 

def prepare_training():
    if (config.get('resume') is not None) and os.path.exists(config.get('resume')):
        sv_file = torch.load(config['resume'],map_location=torch.device('cpu'))
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'],batchsize=config['train_batchsize'], base_lr=config['base_lr'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for e in range(1,epoch_start):
            lr_scheduler.step(e)
            #lr_scheduler.step()
        print("epoch start from: ",epoch_start,"lr: ",optimizer.param_groups[0]['lr'])
    elif(config.get('finetune') is not None) and os.path.exists(config.get('finetune')):
        
        encoder_spec = {}
        encoder_spec["name"] = "edsr-baseline"

        encoder_spec["args"] = {
            "no_upsampling" : True
        }
        model = Classifier(encoder_spec,width=256,num_classes=1000).cuda()
        checkpoint = torch.load(config.get('finetune'),map_location='cpu')
        checkpoint_model = checkpoint['model']['sd']
        decoderKeys = []
        for key in checkpoint_model.keys():
                if(key.startswith('decoder')):
                    decoderKeys.append(key)
        for key in decoderKeys:
            del checkpoint_model[key]

        interpolate_pos_embed(model , checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        trunc_normal_(model.head.weight, std=2e-5)

        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'],batchsize=config['train_batchsize'] , base_lr=config.get('base_lr'))
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler



def train(train_loader, model, optimizer, \
         epoch):

    
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    train_loss = utils.Averager()

    iteration = 0
    pbar = tqdm(train_loader, leave=False, desc='train')


    for batch in pbar:        
        for k,v in batch.items():
            batch[k] = v.cuda(non_blocking=True)
        b , c , h , w = batch["img"].shape
        pred = model(batch["img"])
        target  = batch["gt"]
        
        loss = loss_fn(pred , target)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        iteration += 1
        
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = None; loss = None
        pbar.set_description('train loss: {:.4f}, lr: {:.6f}'.format(train_loss.item(),optimizer.param_groups[0]['lr'] ))

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        train_loss = train(train_loader, model, optimizer, \
                           epoch)
        if lr_scheduler is not None:
            # lr_scheduler.step()
            lr_scheduler.step(epoch)

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        # 这里在干嘛？为什么要新建（？）一个模型
        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if ((epoch_save is not None) and (epoch % epoch_save == 0)):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch == 1) or ((epoch_val is not None) and (epoch % epoch_val == 0)):
            if n_gpus > 1: #and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_finetune(val_loader,model_)

            log_info.append('val: psnr={:.4f}'.format(val_res))
#             writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='./configs/train_edsr-sronet.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0,1')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)

