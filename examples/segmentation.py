"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from Swin3D.models import Swin3DUNet
from easydict import EasyDict
from MinkowskiEngine import SparseTensor
import time

import open3d

from utils.data_util import data_prepare_scannet

# Structured3D semantic segmentation label set
S25_LABEL_SET = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15,\
    16, 17, 18, 19, 22, 24, 25, 32, 34, 35, 38, 39, 40]

# color palette
COLOR_TO_ADEK_LABEL = {
    0: (120, 120, 120), #  1: "wall",
    1: (80, 50, 50), # 2: "floor",
    2: (224, 5, 255), # 3: "cabinet",
    3: (204, 5, 255), # 4: "bed",
    4: (204, 70, 3), # 5: "chair",
    5: (11, 102, 255), # 6: "sofa",
    6: (255, 6, 82), # 7: "table",
    7: (8, 255, 51), # 8: "door",
    8: (230, 230, 230), # 9: "window",
    9: (255, 6, 51), # 11: "picture",
    10: (10, 255, 71), # 14: "desk",
    11: (255, 7, 71), # 15: "shelves",
    12: (255, 51, 8), # 16: "curtain",
    13: (6, 51, 255), # 17: "dresser",
    14: (0, 235, 255), # 18: "pillow",
    15: (220, 220, 220), # 19: "mirror",
    16: (120, 120, 80), # 22: "ceiling",
    17: (20, 255, 0), # 24: "fridge",
    18: (0, 255, 194), # 25: "television",
    19: (146, 111, 194), # 32: "night stand",
    20: (0, 163, 255), # 34: "sink",
    21: (0, 31, 255), # 35: "lamp",
    22: (94, 106, 211), # 38: "structure",
    23: (82, 84, 163), # 39: "furniture",
    24: (100, 85, 144) # 40: "prop",
}
COLOR_LABELS = np.array([COLOR_TO_ADEK_LABEL[i] for i in range(25)], dtype=np.uint8)

args = EasyDict({
    'in_channels': 9,
    'num_layers': 5,
    'depths': [2, 4, 9, 4, 4],
    'channels': [48, 96, 192, 384, 384] ,
    # 'channels': [80, 160, 320, 640, 640],  # Large model
    'num_heads': [6, 6, 12, 24, 24],
    # 'num_heads': [10, 10, 20, 40, 40],  # Large model
    'window_sizes': [5, 7, 7, 7, 7],
    'quant_size': 4,
    'down_stride': 3,
    'knn_down': True,
    'stem_transformer': True,
    'upsample': 'linear',
    'up_k': 3,
    'drop_path_rate': 0.3,
    'num_classes': 25,
    'ignore_label': -100,
    'base_lr': 0.001,
    'transformer_lr_scale': 0.1,
    'weight_decay': 0.0001,
})
model = Swin3DUNet(depths=args.depths, 
                   channels=args.channels, 
                   num_heads=args.num_heads, 
                   window_sizes=args.window_sizes, 
                   quant_size=args.quant_size, 
                   up_k=args.up_k, 
                   drop_path_rate=args.drop_path_rate, 
                   num_classes=args.num_classes, 
                   num_layers=args.num_layers, 
                   stem_transformer=args.stem_transformer, 
                   upsample=args.upsample, 
                   first_down_stride=args.down_stride,
                   knn_down=args.knn_down, 
                   in_channels=args.in_channels, 
                   cRSE='XYZ_RGB_NORM', 
                   fp16_mode=0)
# print(model)
model.load_pretrained_model(ckpt="../pretrained-models/Swin3D_RGBN_S.pth", skip_first_conv=False)
print('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
model = model.cuda()
criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "blocks" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model.named_parameters() if "blocks" in n and p.requires_grad],
        "lr": args.base_lr * args.transformer_lr_scale,
    },
]
optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)

# data = np.load("examples/input.npz")
data_xyz, data_feat, data_labels = torch.load("/data/dataset/Structured3D/swin3d/swin3d/train/scene_00000_485142_1cm_seg.pth")
xyz, feat, target = data_prepare_scannet(coord=data_xyz, 
                                         feat=data_feat, 
                                         label=data_labels, 
                                         split="test", 
                                         voxel_size=0.02, 
                                         voxel_max=None, 
                                         transform=None, 
                                         shuffle_index=False)
batch = torch.zeros((xyz.shape[0],), dtype=torch.int32)
print(f'feat: {feat.shape}')
print(f'xyz: {xyz.shape}')
print(f'batch: {batch.shape}')
print(f'target: {target.shape}')
input_pcl = open3d.geometry.PointCloud()
input_pcl.points = open3d.utility.Vector3dVector(xyz)
input_pcl.colors = open3d.utility.Vector3dVector(feat[:, 0:3])
input_pcl.normals = open3d.utility.Vector3dVector(feat[:, 3:6])
open3d.io.write_point_cloud("input_pcl.ply", input_pcl)
# feats: [N, 6], RGB, Normal
# xyz: [N, 3],
# batch: [N],
# target: [N],
begin_tms = time.time()
# feat, xyz, batch, target = torch.from_numpy(feat).cuda(), torch.from_numpy(xyz).cuda(), torch.from_numpy(batch).cuda(), torch.from_numpy(target).cuda()
feat, xyz, batch, target = feat.cuda(), xyz.cuda(), batch.cuda(), target.cuda()
coords = torch.cat([batch.unsqueeze(-1), xyz], dim=-1)
feat = torch.cat([feat, xyz], dim=1)
print(f'feat: {feat.float().shape} coords: {torch.floor(coords).int().shape}')
sp = SparseTensor(features=feat.float(), coordinates=torch.floor(coords).int(), device=feat.device)
print(f'sp.C: {sp.C.shape}, sp.F: {sp.F.shape}')
colors = feat[:, 0:3]
normals = feat[:, 3:6]
coords_sp = SparseTensor(features=torch.cat([coords, colors, normals], dim=1), coordinate_map_key=sp.coordinate_map_key, 
coordinate_manager=sp.coordinate_manager)

use_amp = False
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast(enabled=use_amp):
    output = model(sp, coords_sp)
loss = criterion(output, target)
print(f'loss: {loss}')
# save output to point cloud
print(f'output: {output.shape}')
end_tms = time.time()
print(f'Elapsed time: {end_tms - begin_tms} seconds')
output_point_labels = output.argmax(dim=1).cpu().numpy()
output_pcl = open3d.geometry.PointCloud()
output_pcl.points = open3d.utility.Vector3dVector(xyz.cpu().numpy())
point_colors = COLOR_LABELS[output_point_labels]
output_pcl.colors = open3d.utility.Vector3dVector(point_colors/255.0)
open3d.io.write_point_cloud("output_pcl.ply", output_pcl)
optimizer.zero_grad()

if use_amp:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
print("FINISHED!")

