import open3d as o3d
import numpy as np
import torch
import math
from models.flowstep3d import FlowStep3D
import yaml
device = torch.device('cuda')
print(device)
cpu = torch.device('cpu')

seq = 9
p1_id = 507
p2_id = 508

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

def are_same_plane(plane1, plane2, tolerance):
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    
    dot_product = a1*a2 + b1*b2 + c1*c2
    if abs(dot_product) < 1 - 0.1:
        return False
    
    dist = abs(d1 - d2) / math.sqrt(a1**2 + b1**2 + c1**2)
    
    if dist <= tolerance:
        return True
    else:
        return False

pc_path = [f'/media/dl/data_pc/data_demo/MotionSeg3D/data/sequences/{seq:02d}/velodyne/scans_{p1_id}.bin',
           f'/media/dl/data_pc/data_demo/MotionSeg3D/data/sequences/{seq:02d}/velodyne/scans_{p2_id}.bin']
poses_path = f'/media/dl/data_pc/data_demo/MotionSeg3D/data/sequences/{seq:02d}/poses.txt'
calib_path = f'/media/dl/data_pc/data_demo/MotionSeg3D/data/sequences/{seq:02d}/calib.txt'

poses = load_poses(poses_path)
calib = load_calib(calib_path)
pc1 = load_vertex(pc_path[0])
pc2 = load_vertex(pc_path[1])

# pc1 = (poses[p1_id] @ calib @ pc1.T).T
# pc2 = (np.linalg.inv(poses[p1_id] @ calib) @ poses[p2_id] @ calib @ pc2.T).T
pc1 = pc1[pc1[:, 2] > -1.5]
pc2 = pc2[pc2[:, 2] > -1.5]

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(pc1[:,:3])

labels = np.asarray(pcd1.cluster_dbscan(eps=0.55, min_points=10))
max_label = max(labels)
print(max_label)
# 循环遍历每个平面
planes_idx = []
planes_arg = []
for i in range(max_label + 1):
    # 获取当前平面的索引
    indices = np.where(labels == i)[0]

    if(len(indices) < 10): continue

    # 根据索引获取当前平面的点云
    cloud = pcd1.select_by_index(indices)
    # 拟合平面
    [a, b, c, d], idx = cloud.segment_plane(0.1, 3, 10)
    planes_idx.append(indices[idx])
    planes_arg.append([a,b,c,d])

for i in range(len(planes_arg)):
    for j in range(i + 1, len(planes_arg)):
        if labels[planes_idx[j][0]] == labels[planes_idx[i][0]]: continue
        if(are_same_plane(planes_arg[i], planes_arg[j], 0.2)):
            labels[planes_idx[j]] = labels[planes_idx[i][0]]


pc1 = torch.tensor(pc1[:,:3]).to(torch.float).unsqueeze(0).to(device)
pc2 = torch.tensor(pc2[:,:3]).to(torch.float).unsqueeze(0).to(device)

config = yaml.safe_load(open('configs/test/flowstep3d_sv.yaml'))
checkpoint = torch.load('/media/dl/data_pc/ckpt/flowstep3d_finetune/2023-03-27_15-38/epoch=18.ckpt')
model = FlowStep3D(**checkpoint["hyper_parameters"])
model_weights = checkpoint["state_dict"]
for key in list(model_weights):
    model_weights[key.replace("model.", "")] = model_weights.pop(key)
model.load_state_dict(model_weights)

model.eval().to(device)
with torch.no_grad():
    sf = model(pc1,pc2,pc1,pc2,10)

sf = sf[-1].cpu().detach().numpy().reshape(-1, 3)
pc1 = pc1.cpu().numpy().reshape(-1, 3)
pc2 = pc2.cpu().numpy().reshape(-1, 3)



pdis = np.sqrt(np.sum(sf ** 2, axis=1))
mean_norm = np.mean(pdis)
print(f'mean norm: {mean_norm}')

label_sf = {}
for i in range(len(labels)):
    if labels[i] not in label_sf:
        label_sf[labels[i]] = [sf[i]]
    else:
        label_sf[labels[i]].append(sf[i])
label_sf_var = np.zeros(max_label + 1)
for label_id in label_sf:
    label_sf_var[label_id] = np.var(label_sf[label_id])

mean_var = np.mean(label_sf_var)
print(f'mean var: {mean_var}')
# pcdm = pc1 + sf
# pcdsf = o3d.geometry.PointCloud()
# pcdsf.points = o3d.utility.Vector3dVector(pcdm[:,:3])
# pcdsf.paint_uniform_color([0,1,0])

mask_mos_idx = pdis > 1.2 * mean_norm
mask_var_idx = label_sf_var < 0.5 * mean_var

# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector(pc2[:,:3])
# pcd2.paint_uniform_color([0,0,1])

colors = []
for i in range(len(pc1)):
    if(mask_mos_idx[i] == 1 and labels[i] >=0 and mask_var_idx[labels[i]] == 1):
        colors.append([1,0,0])
    else:
        colors.append([0,0,1])

# pcd2.colors = o3d.utility.Vector3dVector(colors)
pcd1.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd1])