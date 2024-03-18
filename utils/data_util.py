import numpy as np
import random

from typing import Tuple
import torch
import torchvision
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

def data_prepare_koolai(
    pcl_coords:np.ndarray,
    pcl_feats:np.ndarray,
    pcl_labels:np.ndarray,
    split:str="train",
    voxel_size:float=0.04,
    voxel_max:float=None,
    transform:torchvision.transforms.Compose=None,
    shuffle_index:bool=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ convert point cloud to tensor

    Args:
        pcl_coords (np.ndarray): points coordinates
        pcl_feats (np.ndarray): points features: color, normal
        pcl_labels (np.ndarray): points semantic labels
        split (str, optional): train split. Defaults to "train".
        voxel_size (float, optional): voxel size. Defaults to 0.04.
        voxel_max (float, optional): maxmium voxel size. Defaults to None.
        transform (torchvision.transforms.Compose, optional): input transformation. Defaults to None.
        shuffle_index (bool, optional): whether to shuffle. Defaults to False.

    Returns:
        _type_: _description_
    """
    # transformation
    if transform is not None:
        color = pcl_feats[:, 0:3]
        normal = pcl_feats[:, 3:6]
        if normal.shape[1] == 0:
            normal = None
            pcl_coords, color = transform(pcl_coords, color)
        else:
            pcl_coords, color, normal = transform(pcl_coords, color, normal)
            pcl_feats[:, 3:6] = normal
        pcl_feats[:, 0:3] = color

    # voxelization
    if voxel_size:
        coord_min = np.min(pcl_coords, 0)
        pcl_coords -= coord_min
        pcl_coords = pcl_coords.astype(np.float32)
        uniq_idx = voxelize(pcl_coords, voxel_size)
        pcl_coords, pcl_feats,  = pcl_coords[uniq_idx], pcl_feats[uniq_idx]
        pcl_coords = pcl_coords / voxel_size
        
    if pcl_labels is not None:
        pcl_labels = pcl_labels[uniq_idx]
        # shrink to voxel_max
        if voxel_max and pcl_labels.shape[0] > voxel_max:
            init_idx = (
                np.random.randint(pcl_labels.shape[0])
                if "train" in split
                else pcl_labels.shape[0] // 2
            )
            crop_idx = np.argsort(np.sum(np.square(pcl_coords - pcl_coords[init_idx]), 1))[:voxel_max]
            pcl_coords, pcl_feats, pcl_labels = pcl_coords[crop_idx], pcl_feats[crop_idx], pcl_labels[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(pcl_coords.shape[0])
        np.random.shuffle(shuf_idx)
        pcl_coords, pcl_feats = pcl_coords[shuf_idx], pcl_feats[shuf_idx]
        if pcl_labels is not None:
            pcl_labels = pcl_labels[shuf_idx]

    pcl_coords = torch.FloatTensor(pcl_coords)
    pcl_feats = torch.FloatTensor(pcl_feats)
    if pcl_labels is not None:
        pcl_labels = torch.LongTensor(pcl_labels)
        return pcl_coords, pcl_feats, pcl_labels
    return pcl_coords, pcl_feats

