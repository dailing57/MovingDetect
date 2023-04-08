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
p1_id = 509
p2_id = 510

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

def PCA(data):
    H = np.dot(data.T,data)  #求解协方差矩阵 H
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)    # SVD求解特征值、特征向量
    sort = eigenvalues.argsort()[::-1]      #降序排列
    eigenvalues = eigenvalues[sort]         #索引
    eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

pc_path = [f'/media/dl/data_pc/data_demo/MotionSeg3D/data/sequences/{seq:02d}/velodyne/scans_{p1_id}.bin',
           f'/media/dl/data_pc/data_demo/MotionSeg3D/data/sequences/{seq:02d}/velodyne/scans_{p2_id}.bin']
poses_path = f'/media/dl/data_pc/data_demo/MotionSeg3D/data/sequences/{seq:02d}/poses.txt'
calib_path = f'/media/dl/data_pc/data_demo/MotionSeg3D/data/sequences/{seq:02d}/calib.txt'

poses = load_poses(poses_path)
calib = load_calib(calib_path)


# pc1 = (poses[p1_id] @ calib @ pc1.T).T
# pc2 = (np.linalg.inv(poses[p1_id] @ calib) @ poses[p2_id] @ calib @ pc2.T).T
pc1, pc2 = load_vertex(pc_path[0])[:, :3], load_vertex(pc_path[1])[:, :3]
pcd1, pcd2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
pcd1.points, pcd2.points = o3d.utility.Vector3dVector(pc1), o3d.utility.Vector3dVector(pc2)
pcd1.estimate_normals(), pcd2.estimate_normals()
pc1_normals, pc2_normals = np.array(pcd1.normals), np.array(pcd2.normals)


is_not_ground_s = (pc1[:, 2] > -1.4)
is_not_ground_t = (pc2[:, 2] > -1.4)
pc1 = pc1[is_not_ground_s]
pc1_normals = pc1_normals[is_not_ground_s]
pc2 = pc2[is_not_ground_t]
pc2_normals = pc2_normals[is_not_ground_t]


horizontal_normals_s = np.abs(pc1_normals[:, -1]) < .85
horizontal_normals_t = np.abs(pc2_normals[:, -1]) < .85
pc1 = pc1[horizontal_normals_s]
pc1_normals = pc1_normals[horizontal_normals_s]
pc2 = pc2[horizontal_normals_t]
pc2_normals = pc2_normals[horizontal_normals_t]


is_near_s = (np.amax(np.abs(pc1), axis=1) < 35)
is_near_t = (np.amax(np.abs(pc2), axis=1) < 35)
pc1 = pc1[is_near_s]
pc1_normals = pc1_normals[is_near_s]
pc2 = pc2[is_near_t]
pc2_normals = pc2_normals[is_near_t]

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
checkpoint = torch.load('/media/dl/data_pc/ckpt/flowstep3d_finetune/2023-04-04_15-33/epoch=3.ckpt')
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

label_sf, label_pt = {}, {}
for i in range(len(labels)):
    if labels[i] not in label_sf:
        label_sf[labels[i]] = [sf[i]]
        label_pt[labels[i]] = [pc1[i]]
    else:
        label_sf[labels[i]].append(sf[i])
        label_pt[labels[i]].append(pc1[i])
label_sf_var = np.zeros(max_label + 1)
for label_id in label_sf:
    label_sf_var[label_id] = np.var(label_sf[label_id])

vis = o3d.visualization.Visualizer()
vis.create_window()

mask_horizon = np.ones(max_label + 1)
for label_id in label_sf:
    w, v = PCA(np.array(label_sf[label_id]))
    w = w / np.sum(w)
    v1 = w[0] * v[:, 0]
    v2 = w[1] * v[:, 1]
    v3 = w[2] * v[:, 2]
    main_v = np.abs(v1 + v2 + v3)
    v_h = np.array([main_v[0], main_v[1], 0])
    cos_theta = np.dot(main_v, v_h) / (np.linalg.norm(main_v) * np.linalg.norm(v_h))
    theta = np.arccos(cos_theta) * 180 / np.pi
    if(np.abs(theta) > 75):
        mask_horizon[label_id] = 0
    point = [[0,0,0],v1,v2,v3]
    point += np.mean(label_pt[label_id], axis=0)
    lines = [[0,1], [0,2], [0,3]]
    colors = [[1,0,0], [0.75,0,0], [0.5,0,0]]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)


mean_var = np.mean(label_sf_var)
print(f'mean var: {mean_var}')
# pcdm = pc1 + sf
# pcdsf = o3d.geometry.PointCloud()
# pcdsf.points = o3d.utility.Vector3dVector(pcdm[:,:3])
# pcdsf.paint_uniform_color([0,1,0])

mask_mos_idx = pdis > 1.0 * mean_norm
mask_var_idx = label_sf_var < 0.8 * mean_var
mask_height = pc1[:, 2] <= 5


colors = []
for i in range(len(pc1)):
    if(mask_mos_idx[i] == 1 and labels[i] >=0
       and mask_var_idx[labels[i]] == 1
       and mask_horizon[labels[i]] == 1
       ):
        colors.append([1,0,0])
    else:
        colors.append([0,0,1])

pcd1.colors = o3d.utility.Vector3dVector(colors)
vis.add_geometry(pcd1)
vis.run()
vis.destroy_window()
