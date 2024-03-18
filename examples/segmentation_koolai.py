"""
# Copyright (c) by FANG Chuan.
email: cfangac@connect.ust.hk
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
from PIL import Image
import open3d
from matplotlib import pyplot as plt
from utils.data_util import data_prepare_koolai
from utils.pano_ops import save_color_pointcloud

from icecream import ic
# Structured3D semantic segmentation label set
S25_LABEL_SET = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15,\
    16, 17, 18, 19, 22, 24, 25, 32, 34, 35, 38, 39, 40]
CLASS_TO_LABEL = {
    0: "wall",
    1: "floor",
    2: "cabinet",
    3: "bed",
    4: "chair",
    5: "sofa",
    6: "table",
    7: "door",
    8: "window",
    9: "picture",
    10: "desk",
    11: "shelves",
    12: "curtain",
    13: "dresser",
    14: "pillow",
    15: "mirror",
    16: "ceiling",
    17: "fridge",
    18: "television",
    19: "night stand",
    20: "sink",
    21: "lamp",
    22: "structure",
    23: "furniture",
    24: "prop",
}
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


def prepare_swin3d_input(rgb_pano_filepath:str, 
                         depth_pano_filepath:str, 
                         voxel_size:int = 0.02, 
                         save_input_filepath:str=None, 
                         device:torch.device=torch.device("cuda")) -> (SparseTensor, SparseTensor):
    
    rgb_img = np.array(Image.open(rgb_pano_filepath).convert('RGB'))
    depth_img = np.array(Image.open(depth_pano_filepath))
    
    o3d_pcl = save_color_pointcloud(rgb_img=rgb_img, 
                                    depth_img=depth_img,  
                                    depth_scale=4000.0, 
                                    normaliz=False,
                                    saved_color_pcl_filepath=None,)
    pcl_xyzs = np.array(o3d_pcl.points)
    pcl_colors = np.array(o3d_pcl.colors)
    pcl_normals = np.array(o3d_pcl.normals)
    pcl_feats = np.concatenate([pcl_colors, pcl_normals], axis=1)
    xyzs, feats = data_prepare_koolai(pcl_coords=pcl_xyzs, 
                                            pcl_feats=pcl_feats, 
                                            pcl_labels=None, 
                                            split="test", 
                                            voxel_size=voxel_size, 
                                            voxel_max=None, 
                                            transform=None, 
                                            shuffle_index=False)
    print(f'feat: {feats.shape}')
    print(f'xyzs: {xyzs.shape}')
    
    input_batches = torch.zeros((xyzs.shape[0],), dtype=torch.int32)

    # move them to cuda if available
    input_xyzs, input_feats, input_batches = xyzs.to(device), feats.to(device), input_batches.to(device)

    coords = torch.cat([input_batches.unsqueeze(-1), input_xyzs], dim=-1)
    feat = torch.cat([input_feats, input_xyzs], dim=1)
    input_sp = SparseTensor(features=feat.float(), coordinates=torch.floor(coords).int(), device=feat.device)
    colors = feat[:, 0:3]
    normals = feat[:, 3:6]
    input_coords_sp = SparseTensor(features=torch.cat([coords, colors, normals], dim=1), 
                            coordinate_map_key=input_sp.coordinate_map_key, 
                            coordinate_manager=input_sp.coordinate_manager)
    return input_sp, input_coords_sp

def prepare_swin3d_model(checkpoint_filepath:str='../pretrained-models/Swin3D_RGBN_S.pth',
                         device:torch.device=torch.device("cuda")) -> Swin3DUNet:
    """ load Swin3D model

    Args:
        checkpoint_filepath (str, optional): _description_. Defaults to '../pretrained-models/Swin3D_RGBN_S.pth'.
        device (torch.device, optional): _description_. Defaults to torch.device("cuda").

    Returns:
        Swin3DUNet: _description_
    """
    swin3d_s_default_args = EasyDict({
        'in_channels': 9,
        'num_layers': 5,
        'depths': [2, 4, 9, 4, 4],
        'channels': [48, 96, 192, 384, 384] , # Small model
        'num_heads': [6, 6, 12, 24, 24], # Small model
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
    swin3d_l_default_args = EasyDict({
        'in_channels': 9,
        'num_layers': 5,
        'depths': [2, 4, 9, 4, 4],
        'channels': [80, 160, 320, 640, 640],  # Large model
        'num_heads': [10, 10, 20, 40, 40],  # Large model
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
    model = Swin3DUNet(depths=swin3d_s_default_args.depths, 
                    channels=swin3d_s_default_args.channels, 
                    num_heads=swin3d_s_default_args.num_heads, 
                    window_sizes=swin3d_s_default_args.window_sizes, 
                    quant_size=swin3d_s_default_args.quant_size, 
                    up_k=swin3d_s_default_args.up_k, 
                    drop_path_rate=swin3d_s_default_args.drop_path_rate, 
                    num_classes=swin3d_s_default_args.num_classes, 
                    num_layers=swin3d_s_default_args.num_layers, 
                    stem_transformer=swin3d_s_default_args.stem_transformer, 
                    upsample=swin3d_s_default_args.upsample, 
                    first_down_stride=swin3d_s_default_args.down_stride,
                    knn_down=swin3d_s_default_args.knn_down, 
                    in_channels=swin3d_s_default_args.in_channels, 
                    cRSE='XYZ_RGB_NORM', 
                    fp16_mode=0)
    # print(model)
    model.load_pretrained_model(ckpt=checkpoint_filepath, skip_first_conv=False)
    print('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
    model = model.to(device)
    # param_dicts = [
    #     {"params": [p for n, p in model.named_parameters() if "blocks" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in model.named_parameters() if "blocks" in n and p.requires_grad],
    #         "lr": swin3d_s_default_args.base_lr * swin3d_s_default_args.transformer_lr_scale,
    #     },
    # ]
    # optimizer = torch.optim.AdamW(param_dicts, lr=swin3d_s_default_args.base_lr, weight_decay=swin3d_s_default_args.weight_decay)
    return model


def test_on_koolai_data(input_rgb_pano_filepath:str = '/mnt/nas_3dv/hdd1/datasets/KoolAI/20240229_data/data/3FO4K5FY8CPL/panorama/room_973/0/rgb.png',
                        input_depth_pano_filepath:str = '/mnt/nas_3dv/hdd1/datasets/KoolAI/20240229_data/data/3FO4K5FY8CPL/panorama/room_973/0/depth.png'):
    voxel_size = 0.02

    input_xyzs, input_feats = prepare_swin3d_input(rgb_pano_filepath=input_rgb_pano_filepath,
                                                depth_pano_filepath=input_depth_pano_filepath,
                                                voxel_size=0.02,
                                                save_input_filepath=None)
    # feats: [N, 6], RGB, Normal
    # input_xyzs: [N, 3],
    # batch: [N],
    input_batches = torch.zeros((input_xyzs.shape[0],), dtype=torch.int32)

    # move them to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_xyzs, input_feats, input_batches = input_xyzs.to(device), input_feats.to(device), input_batches.to(device)

    coords = torch.cat([input_batches.unsqueeze(-1), input_xyzs], dim=-1)
    feat = torch.cat([input_feats, input_xyzs], dim=1)
    # print(f'feat: {feat.float().shape} coords: {torch.floor(coords).int().shape}')
    sp = SparseTensor(features=feat.float(), coordinates=torch.floor(coords).int(), device=feat.device)
    # print(f'sp.C: {sp.C.shape}, sp.F: {sp.F.shape}')
    colors = feat[:, 0:3]
    normals = feat[:, 3:6]
    coords_sp = SparseTensor(features=torch.cat([coords, colors, normals], dim=1), 
                            coordinate_map_key=sp.coordinate_map_key, 
                            coordinate_manager=sp.coordinate_manager)

    swin3d_S_ckpt_path = '../pretrained-models/Swin3D_RGBN_S.pth'
    swin3d_L_ckpt_path = '../pretrained-models/Swin3D_RGBN_L.pth'
    model = prepare_swin3d_model(checkpoint_filepath=swin3d_S_ckpt_path, device=device)
    output = model(sp, coords_sp)

    # save output to point cloud
    print(f'output: {output.shape}')
    output_pointabels = output.argmax(dim=1).cpu().numpy()
    output_pcl = open3d.geometry.PointCloud()
    output_pcl.points = open3d.utility.Vector3dVector(input_xyzs.cpu().numpy()*voxel_size)
    point_colors = COLOR_LABELS[output_pointabels]
    output_pcl.colors = open3d.utility.Vector3dVector(point_colors/255.0)
    open3d.io.write_point_cloud("output_pcl.ply", output_pcl)

    # # get bbox of the segmented point cloud
    # predict_labels = np.unique(output_pointabels)
    # print(f'predict_labels: {predict_labels}')
    # for label in predict_labels:
    #     if label == 0:
    #         print("skip wall")
    #         continue
    #     if label == 1:
    #         print("skip floor")
    #         continue
    #     if label == 16:
    #         print("skip ceiling")
    #         continue
    #     mask = output_pointabels == label
    #     object_points = input_xyzs.cpu().numpy()[mask]
    #     object_pcl = output_pcl.select_by_index(np.where(mask)[0])
    #     open3d.io.write_point_cloud(f"object_{CLASS_TO_LABEL[label]}.ply", object_pcl)
    #     # calculate the bounding box by caving voxel grid
        
    #     instace_labels = np.array(object_pcl.cluster_dbscan(eps=0.1, min_points=50, print_progress=True))
    #     # print(f'instace_labels: {instace_labels.shape}')
    #     max_label = instace_labels.max()
    #     print(f"{CLASS_TO_LABEL[label]}  has {max_label + 1} clusters")
    #     colors = plt.get_cmap("tab20")(instace_labels / (max_label if max_label > 0 else 1))
    #     for instace_label in range(max_label + 1):
    #         instance_mask = instace_labels == instace_label
    #         # print(f'instance_mask: {instance_mask.shape}')
    #         instance_pcl = object_pcl.select_by_index(np.where(instance_mask)[0])
    #         open3d.io.write_point_cloud(f"object_{CLASS_TO_LABEL[label]}_{instace_label}.ply", instance_pcl)
    #         # points = np.array(instance_pcl.points)
    #         # bbox = np.array([np.min(points, axis=0), np.max(points, axis=0)])
    #         # print(f'bbox: {bbox}')
    #         # bbox_pcl = open3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[0], max_bound=bbox[1])
    #         bbox = instance_pcl.get_axis_aligned_bounding_box()
    #         import trimesh
    #         box_size = bbox.get_max_bound() - bbox.get_min_bound()
    #         transform_matrix = np.eye(4)
    #         transform_matrix[0:3, 3] = bbox.get_center()
    #         box_trimesh_fmt = trimesh.creation.box(box_size, transform_matrix)
    #         box_trimesh_fmt.export(f"object_{CLASS_TO_LABEL[label]}_{instace_label}_bbox.ply")
    #         # bbox to mesh
    #         # bbox_mesh = open3d.geometry.TriangleMesh.create_from_oriented_bounding_box(bbox).compute_vertex_normals()
    #         # open3d.io.write_triangle_mesh(f"object_{CLASS_TO_LABEL[label]}_{instace_label}_bbox.ply", bbox_mesh)
    #         # open3d.visualization.draw_geometries([input_pcl, output_pcl, bbox_pcl])

from cagroup_roi_head import CAGroup3DRoIHead


# Class Test(nn.Module):
#     def __init__(self):
#         super(Test, self).__init__()
#         self.conv = nn.Conv1d(6, 64, 1)
#         self.fc = nn.Linear(64, 25)
        
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.fc(x)
#         return x
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_rgb_pano_filepath:str = '/mnt/nas_3dv/hdd1/datasets/KoolAI/20240229_data/data/3FO4K5FY8CPL/panorama/room_973/0/rgb.png'
    input_depth_pano_filepath:str = '/mnt/nas_3dv/hdd1/datasets/KoolAI/20240229_data/data/3FO4K5FY8CPL/panorama/room_973/0/depth.png'
    voxel_size = 0.02
    
    swin3d_S_ckpt_path = '../pretrained-models/Swin3D_RGBN_S.pth'
    swin3d_L_ckpt_path = '../pretrained-models/Swin3D_RGBN_L.pth'
    
    ROIHead_deafult_cfg = EasyDict({
        'MIDDLE_FEATURE_SOURCE': [3],
        'NUM_CLASSES': 25,  # object category number in KoolAI dataset
        'CODE_SIZE': 6,    # centroid, size
        'GRID_SIZE': 7 , # sampled grids in each object bbox
        'VOXEL_SIZE': 0.02, # cm, input voxel size
        'COORD_KEY': 2,
        'POOLING_LAYERS_CHANNELS': [[48, 128, 128]],
        'ENLARGE_RATIO': False,
        'SHARED_FC': [256, 256],
        'CLS_FC': [256, 256],
        'REG_FC': [256, 256],
        'DP_RATIO': 0.3,
        'TEST_SCORE_THR': 0.01,
        'TEST_IOU_THR': 0.5,
        'ROI_PER_IMAGE': 128,
        'ROI_FG_RATIO': 0.9,
        'ROI_CONV_KERNEL': 5,
        'ENCODE_SINCOS': False,
        'USE_IOU_LOSS': False,
        'USE_GRID_OFFSET': False,
        'USE_SIMPLE_POOLING': True,
        'USE_CENTER_POOLING': True,
        'LOSS_WEIGHTS': {
            'RCNN_CLS_WEIGHT': 1.0, # no use
            'RCNN_REG_WEIGHT': 1.0 ,# set to 0.5 if use iou loss
            'RCNN_IOU_WEIGHT': 1.0,
            'CODE_WEIGHT': [1., 1., 1., 1., 1., 1.]
            }
    })
    # ROI aggregation head
    roi_agg_head_model = CAGroup3DRoIHead(model_cfg=ROIHead_deafult_cfg)
    roi_agg_head_model.to(device)
    roi_agg_head_model.eval()

    input_sp, input_coords_sp = prepare_swin3d_input(rgb_pano_filepath=input_rgb_pano_filepath,
                                                depth_pano_filepath=input_depth_pano_filepath,
                                                voxel_size=0.02,
                                                save_input_filepath=None,
                                                device=device)

    input_dict = {'sp': input_sp, 'coords_sp': input_coords_sp}

    swin3d_model = prepare_swin3d_model(checkpoint_filepath=swin3d_S_ckpt_path, device=device)
    output_dict = swin3d_model(input_dict)
    output_points_labels = output_dict['output_point_labels']
    ic(output_points_labels.shape)
    
    # save output to point cloud
    input_xyzs = input_sp.C[:, 1:4]
    output_pointabels = output_points_labels.argmax(dim=1).cpu().numpy()
    output_pcl = open3d.geometry.PointCloud()
    output_pcl.points = open3d.utility.Vector3dVector(input_xyzs.cpu().numpy()*voxel_size)
    point_colors = COLOR_LABELS[output_pointabels]
    output_pcl.colors = open3d.utility.Vector3dVector(point_colors/255.0)
    open3d.io.write_point_cloud("output_pcl.ply", output_pcl)

    # get bbox of the segmented point cloud
    output_dict['pred_bbox_list'] = []
    predict_labels = np.unique(output_pointabels)
    for label in predict_labels:
        if label == 0:
            ic("skip wall")
            continue
        if label == 1:
            ic("skip floor")
            continue
        if label == 16:
            ic("skip ceiling")
            continue
        mask = output_pointabels == label
        object_points = input_xyzs.cpu().numpy()[mask]
        object_pcl = output_pcl.select_by_index(np.where(mask)[0])
        open3d.io.write_point_cloud(f"object_{CLASS_TO_LABEL[label]}.ply", object_pcl)
        
        instace_labels = np.array(object_pcl.cluster_dbscan(eps=0.1, min_points=50, print_progress=True))
        # print(f'instace_labels: {instace_labels.shape}')
        max_label = instace_labels.max()
        print(f"{CLASS_TO_LABEL[label]}  has {max_label + 1} clusters")
        for instace_label in range(max_label + 1):
            instance_mask = instace_labels == instace_label
            # print(f'instance_mask: {instance_mask.shape}')
            instance_pcl = object_pcl.select_by_index(np.where(instance_mask)[0])
            open3d.io.write_point_cloud(f"object_{CLASS_TO_LABEL[label]}_{instace_label}.ply", instance_pcl)

            bbox = instance_pcl.get_axis_aligned_bounding_box()
            import trimesh
            box_size = bbox.get_max_bound() - bbox.get_min_bound()
            transform_matrix = np.eye(4)
            transform_matrix[0:3, 3] = bbox.get_center()
            box_trimesh_fmt = trimesh.creation.box(box_size, transform_matrix)
            box_trimesh_fmt.export(f"object_{CLASS_TO_LABEL[label]}_{instace_label}_bbox.ply")
            
            bbox_center_tensor = torch.tensor(bbox.get_center(), dtype=torch.float32, device=device)
            bbox_size_tensor = torch.tensor(box_size, dtype=torch.float32, device=device)
            bbox_z_angle_tensor = torch.tensor([0.0], dtype=torch.float32, device=device)
            bbox = torch.concat([bbox_center_tensor, bbox_size_tensor, bbox_z_angle_tensor], dim=-1)
            output_dict['pred_bbox_list'].append((bbox, torch.tensor([label], dtype=torch.long, device=device)))

    
    ic(len(output_dict['pred_bbox_list']))
    
    roi_agg_head_model(output_dict)
    