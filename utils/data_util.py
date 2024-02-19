import numpy as np
import random

import torch

from utils.voxelize import voxelize, voxelize_and_inverse

def data_prepare_scannet(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        color = feat[:, 0:3]
        normal = feat[:, 3:6]
        if normal.shape[1] == 0:
            normal = None
            coord, color = transform(coord, color)
        else:
            coord, color, normal = transform(coord, color, normal)
            feat[:, 3:6] = normal
        feat[:, 0:3] = color
        # if split=='train':
        #     coord, feat, label = RandomDropout(0.2)(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = coord.astype(np.float32)
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        coord = coord / voxel_size
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = (
            np.random.randint(label.shape[0])
            if "train" in split
            else label.shape[0] // 2
        )
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    # coord_min = np.min(coord, 0)
    # coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label
