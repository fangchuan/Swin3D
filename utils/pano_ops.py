# import os
# import sys

import numpy as np
# import cv2
import open3d as o3d

def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * np.pi


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * np.pi


def get_unit_spherical_map(h:int=512, w:int=1024):
    # h = 512
    # w = 1024
    coorx, coory = np.meshgrid(np.arange(w), np.arange(h))
    us = -np_coorx2u(coorx, w)
    vs = np_coory2v(coory, h)

    X = np.expand_dims(np.cos(vs) * np.sin(us), 2)
    Y = np.expand_dims(np.cos(vs) * np.cos(us), 2)
    Z = np.expand_dims(np.sin(vs), 2)
    unit_map = np.concatenate([X, Y, Z], axis=2)

    return unit_map


def save_color_pointcloud(rgb_img:np.ndarray, depth_img:np.ndarray, saved_color_pcl_filepath:str, depth_scale:float=1000.0, normaliz:bool=False)->o3d.geometry.PointCloud:
    """
    :param rgb_img: rgb panorama image 
    :param depth_img: depth panorama image
    :param depth_scale: depth scale factor
    :param normaliz: whether normalize depth values
    :param saved_color_pcl_filepath: saved color point cloud filepath
    :return: o3d.geometry.PointCloud
    """

    # if len(depth_img.shape) == 3:
    #     depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    assert len(depth_img.shape) == 2, f'depth image shape should be: {(rgb_img.shape[0], rgb_img.shape[1],)}'
    if np.isnan(depth_img.any()) or len(depth_img[depth_img > 0]) == 0:
        print('empyt depth image')
        exit(-1)

    if rgb_img.shape[2] == 4:
        rgb_img = rgb_img[:, :, :3]
    if np.isnan(rgb_img.any()) or len(rgb_img[rgb_img > 0]) == 0:
        print('empyt rgb image')
        exit(-1)
    color = np.clip(rgb_img, 0.0, 255.0) / 255.0

    depth_img = np.expand_dims((depth_img / depth_scale), axis=2)
    max_depth = np.max(depth_img)
    if normaliz:
        depth_img = depth_img / max_depth
        print(f'max depth: {max_depth}')
    pointcloud = depth_img * get_unit_spherical_map()

    o3d_pointcloud = o3d.geometry.PointCloud()
    o3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud.reshape(-1, 3))
    o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
    # downsample point cloud
    # o3d_pointcloud = o3d_pointcloud.voxel_down_sample(voxel_size=0.02)
    # must constrain normals pointing towards camera
    o3d_pointcloud.estimate_normals()
    o3d_pointcloud.orient_normals_towards_camera_location(camera_location=(0, 0, 0))
    # remove outliers
    # cl, ind = o3d_pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # display_inlier_outlier(o3d_pointcloud, ind)
    if saved_color_pcl_filepath is not None:
        o3d.io.write_point_cloud(saved_color_pcl_filepath, o3d_pointcloud)
    return o3d_pointcloud

