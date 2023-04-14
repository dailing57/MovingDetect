import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import open3d as o3d
import numpy as np
import torch
import math
from models.flowstep3d import FlowStep3D
import yaml
import matplotlib.pyplot as plt

cmap = plt.colormaps.get_cmap('rainbow')
def get_color(c):
    return cmap(c)[:3]

def str_to_trans(poses):
    return np.vstack((np.fromstring(poses, dtype=float, sep=' ').reshape(3, 4), [0, 0, 0, 1]))

def load_poses(pose_path):
    poses = []
    with open(pose_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            poses.append(str_to_trans(line))
    return np.array(poses)

def load_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Tr:' in line:
                return str_to_trans(line.replace('Tr:', ''))

def load_vertex(scan_path):
    current_vertex = np.fromfile(scan_path, dtype=np.float32).reshape((-1, 4))
    current_vertex[:,3] = np.ones(current_vertex.shape[0])
    return current_vertex
            
def open_label(filename):
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))

    sem_label = label & 0xFFFF
    label = [1 if i > 250 else 0 for i in sem_label ]
    return label

seq = "08"
frame_id = [598]
for f_id in frame_id:
    str_fid = "%06d"%(f_id)
    print(str_fid)

    scan_path = f'/media/dl/data_pc/semanticKITTI/sequences/{seq}/velodyne/{str_fid}.bin'
    label_path = f'/media/dl/data_pc/semanticKITTI/sequences/{seq}/labels/{str_fid}.label'
    pc = load_vertex(scan_path)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc[:, :3]))
    label = open_label(label_path)
    colors = []
    for i in range(len(label)):
        if label[i] == 1:
            colors.append([1, 0, 0])
        else:
            colors.append([0.25,0.25,0.25])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()



