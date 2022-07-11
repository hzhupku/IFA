import argparse
import os.path as osp
import yaml
import logging
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn

from pyseg.models.model_helper import ModelBuilder

import torch.distributed as dist

from pyseg.utils.loss_helper import get_criterion
from pyseg.utils.lr_helper import get_scheduler, get_optimizer

from pyseg.utils.utils import AverageMeter, intersectionAndUnion, init_log, load_trained_model
from pyseg.utils.utils import set_random_seed, get_world_size, get_rank
from pyseg.dataset.builder import get_loader

parser = argparse.ArgumentParser(description="Pytorch Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument("--local_rank", type=int, default=0)
logger =init_log('global', logging.INFO)
logger.propagate = 0


def main():
    global args, cfg
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        #synchronize()
    rank = get_rank()
    world_size = get_world_size()
    print('rank,world_size',rank,world_size)
    if rank == 0:
        logger.info(cfg)
    if args.seed is not None:
        print('set random seed to',args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg['saver']['snapshot_dir']) and rank == 0:
        os.makedirs(cfg['saver']['snapshot_dir'])
    # Create network.
    model = ModelBuilder(cfg['net'])
    modules_back = [model.encoder]
    modules_head = [model.auxor, model.decoder]
   
    device = torch.device("cuda")
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            find_unused_parameters=True,
        ) 
    if cfg['saver']['pretrain']:
        state_dict = torch.load(cfg['saver']['pretrain'], map_location='cpu')['model_state']
        print("Load trained model from ", str(cfg['saver']['pretrain']))
        load_trained_model(model, state_dict)
    
    #model.cuda()
    if rank ==0:
        logger.info(model)

    criterion = get_criterion(cfg)

    trainloader, valloader = get_loader(cfg)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg['trainer']
    cfg_optim = cfg_trainer['optimizer']

    params_list = []
    for module in modules_back:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim['kwargs']['lr']))
    for module in modules_head:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim['kwargs']['lr']*10))

    optimizer = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(cfg_trainer, len(trainloader), optimizer)  # TODO

    # Start to train model
    best_prec = 0
    for epoch in range(cfg_trainer['epochs']):
        # Training
        train(model, optimizer, lr_scheduler, criterion, trainloader, epoch)
        # Validataion
        if cfg_trainer["eval_on"]:
            if rank ==0:
                logger.info("start evaluation")
            prec = validate(model, valloader, epoch)
            if rank == 0:
                if prec > best_prec:
                    best_prec = prec
                    state = {'epoch': epoch,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict()}
                    torch.save(state, osp.join(cfg['saver']['snapshot_dir'], 'best.pth'))
                    logger.info('Currently, the best val result is: {}'.format(best_prec))
        # note we also save the last epoch checkpoint
        if epoch == (cfg_trainer['epochs'] - 1) and rank == 0:
            state = {'epoch': epoch,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict()}
            torch.save(state, osp.join(cfg['saver']['snapshot_dir'], 'epoch_' + str(epoch) + '.pth'))
            logger.info('Save Checkpoint {}'.format(epoch))
    


def train(model, optimizer, lr_scheduler, criterion, data_loader, epoch):
    model.train()
    data_loader.sampler.set_epoch(epoch)
    num_classes, ignore_label = cfg['net']['num_classes'], cfg['dataset']['ignore_label']
    rank, world_size = get_rank(), get_world_size()

    losses = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        i_iter = epoch * len(data_loader) + step
        lr = lr_scheduler.get_lr()
        lr_scheduler.step()

        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        preds = model(images)
        
        loss = criterion(preds, labels) / world_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get the output produced by model
        output = preds[0] if cfg['net'].get('aux_loss', False) else preds
        output = output.data.max(1)[1].cpu().numpy()
        target = labels.cpu().numpy()
       
        # alculate miou
        intersection, union, target = intersectionAndUnion(output, target, num_classes, ignore_label)

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())

        # gather all loss from different gpus
        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        #print('rank,reduced_loss',rank,reduced_loss)
        losses.update(reduced_loss.item())

        if i_iter % 50 == 0 and rank==0:
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            logger.info('iter = {} of {} completed, LR = {} loss = {}, mIoU = {}'
                        .format(i_iter, cfg['trainer']['epochs']*len(data_loader), lr, losses.avg, mIoU))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    if rank == 0:
        logger.info('=========epoch[{}]=========,Train mIoU = {}'.format(epoch, mIoU))


def validate(model, data_loader, epoch):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = cfg['net']['num_classes'], cfg['dataset']['ignore_label']
    pointrend = 'pointrend' in cfg['net']['decoder']['type']
    rank, world_size = get_rank(), get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()
        with torch.no_grad():
            if not pointrend:
                preds = model(images)
            else:
                preds = model(images, infer=True)

        # get the output produced by model
        output = preds[0] if cfg['net'].get('aux_loss', False) else preds
        output = output.data.max(1)[1].cpu().numpy()
        target = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(output, target, num_classes, ignore_label)

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    
    print(mIoU)
    if rank == 0:
        logger.info('=========epoch[{}]=========,Val mIoU = {}'.format(epoch, mIoU))
    torch.save(mIoU, 'eval_metric.pth.tar')
    return mIoU


if __name__ == '__main__':
    main()