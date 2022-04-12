import torch
import os
import sys
import numpy as np
from PIL import Image
from torch.utils import data
import glob2
import pickle
import os.path
import torch.nn.functional as F
from utils import pad_tensor, pad_list_tensors, create_mask

num_classes = 239

def make_dataset(mode, val=-1):
    video_path = root + 'fcvid_' + mode + '.txt'
    vf = open(video_path, 'r')
    items = vf.readlines()
    vf.close()
    return items


def collate(batch):
    """
    A collate function for data loading with Matterport3D Simulator
    Return: padded tensors
    """
    # unpack into a list
    data = list(zip(*batch))

    padded_vid_feat, vid_length = pad_list_tensors(data[0])
    max_length = max(vid_length)
    padded_vid_feat_mask = create_mask(len(vid_length), 129, vid_length)

    label = list(data[1])
    label = torch.LongTensor(label)

    vid_id = []
    for _vid_id in data[2]:
        vid_id.append(_vid_id)

    if len(data) > 3:
        padded_vid_feat_small, _ = pad_list_tensors(data[3])
        return (padded_vid_feat, padded_vid_feat_mask, padded_vid_feat_small), label, vid_id
    else:
        return padded_vid_feat, label, vid_id


class FCVID(data.Dataset):
    def __init__(self, mode, num_frames=24, frame_caps=30, val=-1, transform=None, small=False, eval=False):
        gt_path = './feas/fcvid/{}_gt.pth'.format(mode)
        self.fea_root = './features/'
        self.small_fea_root = './features_mobile/'

        self.labels = torch.load(gt_path)['gt']
        self.videos = list(self.labels.keys())

        self.frame_caps = frame_caps
        self.num_frames = num_frames
        self.mode = mode
        self.small = small
        self.eval = eval
        if len(self.videos) == 0:
            raise (RuntimeError('Found 0 videos, please check the data set'))

    def __getitem__(self, index):
        vid_name = self.videos[index]

        feature_path = self.fea_root + vid_name + '.pth'
        small_feature_path = self.small_fea_root + vid_name + '.pth'

        frame_features = torch.load(feature_path)['fea']
        frame_features = F.normalize(frame_features, p=2, dim=1)

        label = self.labels[vid_name]
        label = np.argmax(label).item()

        labels = torch.zeros(1, 1).long()
        labels.fill_(label)
        labels = labels.squeeze()

        if self.small:
            small_frame_features = torch.load(small_feature_path)['fea']
            small_frame_features = F.normalize(small_frame_features, p=2, dim=1)
            return frame_features, labels, vid_name, small_frame_features
        else:
            return frame_features, labels, vid_name

    def __len__(self):
        return len(self.videos)