import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
from tqdm import tqdm

from models.lite_eval import LiteEval
from ptflops import profile, flops_to_string
from torch.utils.data import DataLoader
from datasets import fcvid as dset
import torch.nn.functional as F
import math
from utils import cal_map

parser = argparse.ArgumentParser(description='PyTorch LiteEval Test Speed')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--num-frames', default=129, type=float,
                    help='crop size')
parser.add_argument('-large_cell_size', type=int, default=2048, help='hidden size of large rnn cell')
parser.add_argument('-small_cell_size', type=int, default=512, help='hidden size of small rnn call')
parser.add_argument('-gamma', type=float, default=0.85, help='gamma regularization parameter')
parser.add_argument('-tau', type=float, default=0.5, help='temperature')
parser.add_argument('--exp-name', default='experiments_', type=str,
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
    # cuDnn configurations

    global args, best_prec1
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    print("......Speed testing......")  
    model = LiteEval(args, num_classes=dset.num_classes)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    model = model.cuda()
    model.eval()

    test_set = dset.FCVID('test', num_frames=args.num_frames, transform=None, small=True)
    val_loader = DataLoader(test_set, batch_size=args.batch_size,
                                 collate_fn=dset.collate, num_workers=args.workers, shuffle=False)


    model_ops = AverageMeter()
    skim_ratio = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    all_result = []
    all_vids = []
    all_targets = []

    with torch.no_grad():
        for i, (data, target, vid) in enumerate(val_loader):
            inputs = data[0].cuda()
            mask = data[1].cuda()
            inputs_small = data[2].cuda()

            target = target.view(-1)
            target = target.cuda(non_blocking=True)

            # compute output
            lstm_ops, total_params, output, r_stack = profile(model, (inputs, inputs_small), \
                                              print_per_layer_stat=0, input_constructor=True)

            if r_stack.dim() == 3:
                num_skips = r_stack[:, :, 1].sum(0)
            else:
                num_skips = r_stack[:, 1].sum(0)

            # 0.08 GLOPs for MobileNetv2, 7.82 FLOPs for ResNet101
            vid_ops = lstm_ops/10.**9 + args.num_frames*0.08 + (args.num_frames-num_skips)*7.82
            vid_ops = vid_ops.mean().cpu().numpy()

            num_skips = num_skips / r_stack.size(0)
            num_skips = num_skips.mean()
            cur_skim_ratio = float(num_skips.data.cpu())

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            model_ops.update(vid_ops, inputs.size(0))
            skim_ratio.update(cur_skim_ratio, inputs.size(0))

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'SKips {skips.avg:.3f}\t'
                      'Avg GFLOPs {gflops.avg:.3f}'.format(
                    i, len(val_loader), top1=top1, skips=skim_ratio, gflops=model_ops))

            all_result.append(output.data.cpu())
            all_vids.extend(vid)
            all_targets.append(target.data.cpu())

        all_result = torch.cat(all_result, 0)
        all_targets = torch.cat(all_targets, 0)
        top_acc, ap = cal_map(all_result, all_targets)

        print('Top Accuracy: %.2f' % top_acc)
        print('Mean Average Precision: %.2f' % (float(ap) * 100.0))
        print('Average FLOPS: %.2f' % model_ops.avg)
        print('Average Frames Used: %d' % args.num_frames)
        print('Skipping Ratio: %.2f' % skim_ratio.avg)


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