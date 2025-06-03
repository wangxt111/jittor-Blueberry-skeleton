import os
import shutil
import argparse
import numpy as np
import jittor as jt
from jittor import nn
from models.GCN import JointPredNet
from dataset.dataset import get_dataloader, transform
from utils.io_utils import output_point_cloud_ply
from utils.log_utils import AverageMeter
from utils.os_utils import mkdir_p, isdir
from models.supplemental_layers.jittor_chamfer_dist import chamfer_distance_with_average 
from dataset.sampler import SamplerMix

jt.flags.use_cuda = 1 if jt.has_cuda else 0


def save_checkpoint(model, optimizer, epoch, lowest_loss, is_best, checkpoint='checkpoint'):
    jt.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'lowest_loss': lowest_loss
    }, os.path.join(checkpoint, 'checkpoint.pkl'))
    if is_best:
        shutil.copyfile(os.path.join(checkpoint, 'checkpoint.pkl'),
                        os.path.join(checkpoint, 'model_best.pkl'))


def train(train_loader, model, optimizer, args):
    model.train()
    loss_meter = AverageMeter()
    for batch in train_loader:
        data = batch
        optimizer.zero_grad()

        if args.arch == 'masknet':
            mask_pred = model(data)
            mask_gt = data.mask.unsqueeze(1).float()
            loss = nn.binary_cross_entropy_with_logits(mask_pred, mask_gt)
        else:  # jointnet
            displacement = model(data)
            y_pred = displacement + data.poss
            loss = 0.0
            for i in range(len(np.unique(data.joints_batch.numpy()))):
                gt = data.joints[data.joints_batch == i]
                pred = y_pred[data.batch == i]
                loss += chamfer_distance_with_average(pred.unsqueeze(0), gt.unsqueeze(0))
            loss /= args.train_batch

        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

    return loss_meter.avg


def test(test_loader, model, args, save_result=False, best_epoch=None):
    model.eval()
    loss_meter = AverageMeter()
    outdir = args.checkpoint.split('/')[-1]

    for batch in test_loader:
        data = batch

        with jt.no_grad():
            if args.arch == 'masknet':
                mask_pred = model(data)
                mask_gt = data.mask.unsqueeze(1).float()
                loss = nn.binary_cross_entropy_with_logits(mask_pred, mask_gt)
            else:
                displacement = model(data)
                y_pred = displacement + data.pos
                loss = 0.0
                for i in range(len(np.unique(data.joints_batch.numpy()))):
                    gt = data.joints[data.joints_batch == i]
                    pred = y_pred[data.batch == i]
                    loss += chamfer_distance_with_average(pred.unsqueeze(0), gt.unsqueeze(0))
                loss /= args.test_batch
            loss_meter.update(loss.item())

        if save_result:
            output_folder = f'results/{outdir}/best_{best_epoch}/'
            mkdir_p(output_folder)
            if args.arch == 'masknet':
                pred_np = nn.sigmoid(mask_pred).numpy()
                for i in range(len(np.unique(data.batch.numpy()))):
                    np.save(os.path.join(output_folder, f'{data.name[i].item()}_attn.npy'), pred_np[data.batch == i])
            else:
                for i in range(len(np.unique(data.batch.numpy()))):
                    pred_i = y_pred[data.batch == i]
                    output_point_cloud_ply(pred_i, name=str(data.name[i].item()), output_folder=output_folder)

    return loss_meter.avg


def main(args):
    lowest_loss = 1e20

    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    if not args.resume and isdir(args.logdir):
        shutil.rmtree(args.logdir)
    mkdir_p(args.logdir)

    model = JointPredNet(out_channels=3 if args.arch == 'jointnet' else 1,
                         input_normal=args.input_normal, arch=args.arch, aggr=args.aggr)
    
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)

    optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume and os.path.exists(args.resume):
        checkpoint = jt.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']
        lowest_loss = checkpoint['lowest_loss']
        print(f"=> Resumed from checkpoint at epoch {args.start_epoch}")

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

    if args.evaluate:
        print("Evaluation only")
        # test_loss = test(test_loader, model, args, save_result=True, best_epoch=args.start_epoch)
        # print(f'Test Loss: {test_loss:.6f}')
        return

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}")
        train_loss = train(train_loader, model, optimizer, args)
        val_loss = test(val_loader, model, args)
        # test_loss = test(test_loader, model, args)

        print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        is_best = val_loss < lowest_loss
        lowest_loss = min(val_loss, lowest_loss)

        save_checkpoint(model, optimizer, epoch + 1, lowest_loss, is_best, checkpoint=args.checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='masknet')  # jointnet, masknet
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--input_normal', action='store_true')
    parser.add_argument('--aggr', default='max')
    parser.add_argument('--train_batch', default=2, type=int)
    parser.add_argument('--test_batch', default=2, type=int)
    parser.add_argument('--checkpoint', default='checkpoints/test')
    parser.add_argument('--logdir', default='logs/test')
    parser.add_argument('--resume', default='')
    parser.add_argument('--train_folder', type=str)
    parser.add_argument('--val_folder', type=str)
    parser.add_argument('--test_folder', type=str)
    args = parser.parse_args()

    main(args)
