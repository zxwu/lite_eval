import argparse
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils import data
from torch.optim.lr_scheduler import MultiStepLR
import math
from datasets import fcvid as dset
from models.lite_eval import LiteEval

import math
from torchnet import meter
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils import cal_map

parser = argparse.ArgumentParser(description='PyTorch LiteEval')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--forced', dest='forced', action='store_true',
                    help='disable test time pool for DPN models')
parser.add_argument('--num-frames', default=129, type=int,
                    help='crop size')
parser.add_argument('-large_cell_size', type=int, default=2048, help='hidden size of large rnn cell')
parser.add_argument('-small_cell_size', type=int, default=512, help='hidden size of small rnn call')
parser.add_argument('-gamma', type=float, default=0.9, help='gamma regularization parameter')
parser.add_argument('-tau', type=float, default=0.5, help='temperature')
parser.add_argument('--exp-name', default='', type=str,
                    help='name of the experiment. \
                        It decides where to store samples and models')
best_prec1 = 0


def my_collate(batch):
    batch = filter(lambda x: x is not None, batch)
    return default_collate(batch)


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def main():
    global args, best_prec1
    args = parser.parse_args()
    ckpt_path = os.path.join('./experiments', args.exp_name)
    check_mkdir(ckpt_path)

    num_classes = dset.num_classes

    model = LiteEval(args, num_classes=num_classes)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    penalty = nn.MSELoss().cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay)


    cudnn.benchmark = True

    train_set = dset.FCVID('train', num_frames=args.num_frames, small=True)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size,
                                   collate_fn=dset.collate, num_workers=args.workers, shuffle=True)

    test_set = dset.FCVID('test', num_frames=args.num_frames, small=True)
    val_loader = data.DataLoader(test_set, batch_size=args.batch_size,
                                  collate_fn=dset.collate, num_workers=args.workers, shuffle=False)

    scheduler = MultiStepLR(optimizer, milestones=[40, 80])

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    global_step = args.start_epoch*int(len(train_loader)/args.batch_size)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        train(train_loader, model, criterion, penalty, optimizer, epoch, global_step)

        if epoch % 1 == 0:
            prec1, sRatio = validate(val_loader, model, criterion)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint(epoch, prec1, sRatio, model, ckpt_path=ckpt_path)


def train(train_loader, model, criterion, penalty, optimizer, epoch, global_step):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    r = 1e-4
    end = time.time()
    for i, (data, target, vid) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.view(-1)
        inputs = data[0].cuda()
        mask = data[1].cuda()
        inputs_small = data[2].cuda()

        target = target.cuda(non_blocking=True)
        # compute output
        tau = max(0.5, math.exp(-r * global_step))
        output, r_stack = model(inputs, inputs_small, tau)
        if r_stack.dim() == 3:
            num_skips = r_stack[:, :, 1].sum(0)
        else:
            num_skips = r_stack[:, 1].sum(0)

        num_skips = num_skips / r_stack.size(0)
        num_skips = num_skips.mean()

        mse_loss = penalty(num_skips, torch.zeros(1, ).fill_(args.gamma).cuda())
        temp = 1 - math.exp(-global_step / args.epochs) + 1

        loss = criterion(output, target) + temp*mse_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        global_step += 1

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    all_result = []
    all_vids = []
    skim_ratio = []
    all_targets = []
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (data, target, vid) in tqdm(enumerate(val_loader), total=len(val_loader)):

            inputs = data[0].cuda()
            mask = data[1].cuda()
            inputs_small = data[2].cuda()

            target = target.view(-1)
            target = target.cuda(non_blocking=True)

            # compute output
            output, r_stack = model(inputs, inputs_small)

            if r_stack.dim() == 3:
                num_skips = r_stack[:, :, 1].sum(0)
            else:
                r_stack = r_stack.unsqueeze(0)
                num_skips = r_stack[:, :, 1].sum(0)

            num_skips = num_skips/r_stack.size(0)
            num_skips = num_skips.mean()
            cur_skim_ratio = float(num_skips.data.cpu())
            skim_ratio.append(cur_skim_ratio)

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            all_result.append(output.data.cpu())
            all_vids.extend(vid)
            all_targets.append(target.data.cpu())

    all_result = torch.cat(all_result, 0)
    all_targets = torch.cat(all_targets, 0)

    top_acc, ap = cal_map(all_result, all_targets)
    sRatio = sum(skim_ratio) / len(skim_ratio)

    print(' * Prec@1 {top1:.3f} mAP {mAP:.3f} skimRatio {sRatio: .3f}'
          .format(top1=top_acc, mAP=ap, sRatio=sRatio))

    return ap, sRatio


def save_checkpoint(epoch, acc, sRatio, net, ckpt_path):
    snapshot_name = 'epoch_%d_mAP_%.5f_sRatio_%.3f' % (epoch, acc, sRatio)
    torch.save(net.state_dict(), os.path.join(ckpt_path, snapshot_name + '.pth'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()