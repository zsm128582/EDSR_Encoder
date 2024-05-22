import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from test import eval_psnr
from test import eval_randomN
from scheduler import GradualWarmupScheduler
import time
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler


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
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if (config.get('resume') is not None) and os.path.exists(config.get('resume')):
        sv_file = torch.load(config['resume'],map_location=torch.device('cpu'))
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        optimizer.param_groups[0]['lr'] = config['optimizer']['args']['lr']
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for e in range(1,epoch_start):
            lr_scheduler.step(e)
            #lr_scheduler.step()
        print(epoch_start,optimizer.param_groups[0]['lr'])
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

# def trace_handler(p):
#     output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
#     print(output)
#     p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

def train(train_loader, model, optimizer, \
         epoch):
    # print(model)
    # exit()
    
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()
    # metric_fn = utils.calc_psnr

    # inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    # inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    # gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    # gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    #num_dataset = 800 # DIV2K
    #iter_per_epoch = int(num_dataset / config.get('train_dataset')['batch_size'] \
    #                    * config.get('train_dataset')['dataset']['args']['repeat'])
    iteration = 0
    pbar = tqdm(train_loader, leave=False, desc='train')


    for batch in pbar:
        # batch["img"] = batch["img"].cuda(non_blocking=True)
        # batch["coord"] = batch["coord"].cuda(non_blocking = True)
        # batch[]
        # batch["gt"] = batch["gt"].cuda(non_blocking=True)
        # data_load_start = time.time()

        

        for k,v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        # data_load_end = time.time()

        # with torch.profiler.profile(
        # activities=[torch.profiler.ProfilerActivity.CPU , torch.profiler.ProfilerActivity.CUDA],
        # # schedule=torch.profiler.schedule(
        # #     wait=1,
        # #     warmup=2,
        # #     active=6,
        # #     repeat=1),
        # on_trace_ready=torch.profiler.tensorboard_trace_handler("./result"),
        # with_stack=True,
        # record_shapes=False,
        # profile_memory=True
        # ) as profiler:
        # inp = (batch['inp'] - inp_sub) / inp_div
        # forward_start = torch.cuda.Event(enable_timing=True)
        # forward_end = torch.cuda.Event(enable_timing=True)
        # forward_start.record()


        pred = model(batch["img"], batch['coord'])

        # profiler.step()
        # exit()
        # gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, batch["img"])

        # forward_end.record()
        # torch.cuda.synchronize()
        # forward_time = forward_start.elapsed_time(forward_end)
        #psnr = metric_fn(pred, gt)
        
        # tensorboard
        #writer.add_scalars('loss', {'train': loss.item()}, (epoch-1)*iter_per_epoch + iteration)
        #writer.add_scalars('psnr', {'train': psnr}, (epoch-1)*iter_per_epoch + iteration)
        iteration += 1
        
        train_loss.add(loss.item())
        # backward_start = torch.cuda.Event(enable_timing=True)
        # backward_end = torch.cuda.Event(enable_timing=True)
        # backward_start.record()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # backward_end.record()
        # torch.cuda.synchronize()
        # backward_time = backward_start.elapsed_time(backward_end)

        # print(f"For this batch : Data Load Time: {data_load_end - data_load_start:.4f}s, Forward Time: {forward_time:.4f}ms, Backward Time: {backward_time:.4f}ms")
        
        pred = None; loss = None
        pbar.set_description('train {:.4f}'.format(train_loss.item()))  
        # profiler.step()
        # exit()

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

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
            val_res = eval_randomN(val_loader,model)

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

