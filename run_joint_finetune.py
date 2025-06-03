import sys
sys.path.append("./")
import os
import numpy as np
import shutil
import argparse

from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from utils.io_utils import output_point_cloud_ply

import jittor as jt
import jittor.nn as nn
import jittor.nn.functional as F
from jittor.dataset import Dataset, DataLoader

from models.GCN import JOINTNET_MASKNET_MEANSHIFT
from dataset.dataset import get_dataloader, transform
from models.supplemental_layers.jittor_chamfer_dist import chamfer_distance_with_average 
from dataset.sampler import SamplerMix

device = jt.device("cuda" if jt.has_cuda else "cpu")


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    jt.save(state, filepath)  # jittor保存模型用 jt.save

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def pairwise_distances(x, y):
    x_norm = (x ** 2).sum(1).reshape(-1, 1)
    y_t = y.transpose(0, 1)
    y_norm = (y ** 2).sum(1).reshape(1, -1)
    dist = x_norm + y_norm - 2.0 * jt.matmul(x, y_t)
    return jt.clamp(dist, 0.0, float('inf'))


def meanshift_cluster(pts, bandwidth, weights, args):
    pts_steps = []
    for i in range(args.meanshift_step):
        Y = pairwise_distances(pts, pts)
        K = F.relu(bandwidth ** 2 - Y)
        if weights is not None:
            K = K * weights
        P = F.normalize(K, p=1, dim=0, eps=1e-10)
        P = P.transpose(0, 1)
        pts = args.step_size * (jt.matmul(P, pts) - pts) + pts
        pts_steps.append(pts)
    return pts_steps


def main(args):
    global device
    lowest_loss = 1e20

    if not isdir(args.checkpoint):
        print("Create new checkpoint folder " + args.checkpoint)
    mkdir_p(args.checkpoint)
    if not args.resume:
        if isdir(args.logdir):
            shutil.rmtree(args.logdir)
        mkdir_p(args.logdir)

    model = JOINTNET_MASKNET_MEANSHIFT()
    model.to(device)

    optimizer = jt.optim.Adam([
        {'params': model.jointnet.parameters(), 'lr': args.jointnet_lr},
        {'params': model.masknet.parameters(), 'lr': args.masknet_lr},
        {'params': model.bandwidth, 'lr': args.bandwidth_lr}
    ], weight_decay=args.weight_decay)

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = jt.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        pretrained_masknet = jt.load(args.masknet_resume)
        pretrained_jointnet = jt.load(args.jointnet_resume)
        model.masknet.load_state_dict(pretrained_masknet['state_dict'])
        model.jointnet.load_state_dict(pretrained_jointnet['state_dict'])

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))
    
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
        transform=transform,
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            transform=transform,
        )
    else:
        val_loader = None

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

    if args.evaluate:
        print('\nEvaluation only')
        test_loss = test(test_loader, model, args, save_result=True, best_epoch=args.start_epoch)
        print('test_loss {:8f}'.format(test_loss))
        return

    scheduler = jt.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
    
    # jittor不直接支持tensorboard，这里保留接口，如你需要可用其他工具
    from torch.utils.tensorboard import SummaryWriter
    logger = SummaryWriter(log_dir=args.logdir)

    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch: %d ' % (epoch + 1))
        train_loss = train(train_loader, model, optimizer, args)
        val_loss = test(val_loader, model, args)
        test_loss = test(test_loader, model, args)
        scheduler.step()
        print('Epoch{:d}. train_loss: {:.6f}.'.format(epoch + 1, train_loss))
        print('Epoch{:d}. val_loss: {:.6f}.'.format(epoch + 1, val_loss))
        print('Epoch{:d}. test_loss: {:.6f}.'.format(epoch + 1, test_loss))

        is_best = val_loss < lowest_loss
        lowest_loss = min(val_loss, lowest_loss)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'lowest_loss': lowest_loss, 'optimizer': optimizer.state_dict()},
                        is_best, checkpoint=args.checkpoint)

        info = {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}
        for tag, value in info.items():
            logger.add_scalar(tag, value, epoch+1)

    print("=> loading checkpoint '{}'".format(os.path.join(args.checkpoint, 'model_best.pth.tar')))
    checkpoint = jt.load(os.path.join(args.checkpoint, 'model_best.pth.tar'))
    best_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(os.path.join(args.checkpoint, 'model_best.pth.tar'), best_epoch))
    test_loss = test(test_loader, model, args, save_result=True, best_epoch=best_epoch)
    print('Best epoch:\n test_loss {:8f}'.format(test_loss))


def train(train_loader, model, optimizer, args):
    global device
    model.train()
    loss_meter = AverageMeter()
    for data in train_loader:
        # jittor DataLoader 返回的数据可能需要根据你的Dataset自定义转换为jt.array
        data = data.to(device)  # 确保data为jt.array，或自行转换

        optimizer.zero_grad()
        data_displacement, mask_pred_nosigmoid, mask_pred, bandwidth = model(data)
        y_pred = data_displacement + data.pos
        loss_total = 0.0

        batch_unique = jt.unique(data.batch)
        for i in range(len(batch_unique)):
            joint_gt = data.joints[data.joints_batch == i, :]
            y_pred_i = y_pred[data.batch == i, :]
            mask_pred_i = mask_pred[data.batch == i]
            loss_total += chamfer_distance_with_average(y_pred_i.unsqueeze(0), joint_gt.unsqueeze(0))
            clustered_pred = meanshift_cluster(y_pred_i, bandwidth, mask_pred_i, args)
            loss_ms = 0.0
            for j in range(args.meanshift_step):
                loss_ms += chamfer_distance_with_average(clustered_pred[j].unsqueeze(0), joint_gt.unsqueeze(0))
            loss_total = loss_total + args.ms_loss_weight * loss_ms / args.meanshift_step

        loss_total /= len(batch_unique)

        if args.use_bce:
            mask_gt = data.mask.unsqueeze(1)
            loss_total += args.bce_loss_weight * F.binary_cross_entropy_with_logits(mask_pred_nosigmoid, mask_gt.float(), reduction='mean')

        loss_total.backward()
        optimizer.step()
        loss_meter.update(loss_total.item())
    return loss_meter.avg


def test(test_loader, model, args, save_result=False, best_epoch=None):
    global device
    model.eval()
    loss_meter = AverageMeter()
    outdir = args.checkpoint.split('/')[-1]
    for data in test_loader:
        data = data.to(device)
        with jt.no_grad():
            data_displacement, mask_pred_nosigmoid, mask_pred, bandwidth = model(data)
            y_pred = data_displacement + data.pos
            loss_total = 0.0
            batch_unique = jt.unique(data.batch)
            for i in range(len(batch_unique)):
                joint_gt = data.joints[data.joints_batch == i, :]
                y_pred_i = y_pred[data.batch == i, :]
                mask_pred_i = mask_pred[data.batch == i]
                loss_total += chamfer_distance_with_average(y_pred_i.unsqueeze(0), joint_gt.unsqueeze(0))
                clustered_pred = meanshift_cluster(y_pred_i, bandwidth, mask_pred_i, args)
                loss_ms = 0.0
                for j in range(args.meanshift_step):
                    loss_ms += chamfer_distance_with_average(clustered_pred[j].unsqueeze(0), joint_gt.unsqueeze(0))
                loss_total = loss_total + args.ms_loss_weight * loss_ms / args.meanshift_step

            loss_total /= len(batch_unique)
            if args.use_bce:
                mask_gt = data.mask.unsqueeze(1)
                loss_total += args.bce_loss_weight * F.binary_cross_entropy_with_logits(mask_pred_nosigmoid, mask_gt.float(), reduction='mean')

            loss_meter.update(loss_total.item())

            if save_result:
                # 保存点云文件或其他推理结果，代码保持不变
                for bidx in jt.unique(data.batch):
                    pointcloud = data.pos[data.batch == bidx]
                    pred = y_pred[data.batch == bidx]
                    mask = mask_pred[data.batch == bidx]
                    output_point_cloud_ply(os.path.join(args.output_folder, outdir, f"test_{bidx}_{best_epoch}.ply"), pred.detach().cpu().numpy())
    return loss_meter.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Jittor Skeleton Clustering Model')
    parser.add_argument('--train_folder', default='', type=str)
    parser.add_argument('--val_folder', default='', type=str)
    parser.add_argument('--test_folder', default='', type=str)
    parser.add_argument('--checkpoint', default='checkpoint', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--masknet_resume', default='', type=str)
    parser.add_argument('--jointnet_resume', default='', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train_batch', default=12, type=int)
    parser.add_argument('--test_batch', default=4, type=int)
    parser.add_argument('--jointnet_lr', default=1e-4, type=float)
    parser.add_argument('--masknet_lr', default=1e-3, type=float)
    parser.add_argument('--bandwidth_lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--meanshift_step', default=5, type=int)
    parser.add_argument('--step_size', default=0.5, type=float)
    parser.add_argument('--ms_loss_weight', default=0.1, type=float)
    parser.add_argument('--use_bce', action='store_true')
    parser.add_argument('--bce_loss_weight', default=0.1, type=float)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--schedule', nargs='+', default=[30, 60, 90], type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--output_folder', default='outputs', type=str)

    args = parser.parse_args()
    main(args)
