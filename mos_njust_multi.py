import open3d as o3d
import numpy as np
import torch
import math
from models.flowstep3d import FlowStep3D
import yaml

device = torch.device('cuda')

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

def pre_process(pc, pc_normals):
    is_not_ground = (pc[:, 2] > -1.4)
    pc = pc[is_not_ground]
    pc_normals = pc_normals[is_not_ground]

    horizontal_normals = np.abs(pc_normals[:, -1]) < .85
    pc = pc[horizontal_normals]

    is_near = (np.amax(np.abs(pc), axis=1) < 35)
    pc = pc[is_near]

    return pc

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
    
def cluster(pcd):
    labels = np.asarray(pcd.cluster_dbscan(eps=0.55, min_points=10))
    max_label = max(labels)
    print(max_label)
    planes_idx = []
    planes_arg = []
    for i in range(max_label + 1):
        # 获取当前平面的索引
        indices = np.where(labels == i)[0]

        if(len(indices) < 10): continue

        # 根据索引获取当前平面的点云
        cloud = pcd.select_by_index(indices)

        # 拟合平面
        [a, b, c, d], idx = cloud.segment_plane(0.1, 3, 10)
        planes_idx.append(indices[idx])
        planes_arg.append([a,b,c,d])

    for i in range(len(planes_arg)):
        for j in range(i + 1, len(planes_arg)):
            if labels[planes_idx[j][0]] == labels[planes_idx[i][0]]: continue
            if(are_same_plane(planes_arg[i], planes_arg[j], 0.2)):
                labels[planes_idx[j]] = labels[planes_idx[i][0]]
    return labels

def flow_infer(pc1_in, pc2_in):
    pc1 = torch.tensor(pc1_in[:,:3]).to(torch.float).unsqueeze(0).to(device)
    pc2 = torch.tensor(pc2_in[:,:3]).to(torch.float).unsqueeze(0).to(device)

    checkpoint = torch.load('/media/dl/data_pc/ckpt/flowstep3d_finetune/2023-04-04_15-33/epoch=3.ckpt')
    model = FlowStep3D(**checkpoint["hyper_parameters"])
    model_weights = checkpoint["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    model.load_state_dict(model_weights)

    model.eval().to(device)
    with torch.no_grad():
        sf = model(pc1,pc2,pc1,pc2,10)

    return sf[-1].cpu().detach().numpy().reshape(-1, 3)

def gather_pt(sf, labels, pc1):
    label_sf, label_pt = {}, {}
    for i in range(len(labels)):
        if labels[i] not in label_sf:
            label_sf[labels[i]] = [sf[i]]
            label_pt[labels[i]] = [pc1[i]]
        else:
            label_sf[labels[i]].append(sf[i])
            label_pt[labels[i]].append(pc1[i])

    return label_sf, label_pt

def PCA(data):
    H = np.dot(data.T,data)
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)
    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

def gen_horizon_mos_mask(label_sf, horizon_threshold):
    mask_horizon = np.ones(max(label_sf.keys()) + 1)
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
        if(np.abs(theta) > horizon_threshold):
            mask_horizon[label_id] = 0
    return mask_horizon

def gen_var_mask(label_sf, var_threshold):
    label_sf_var = np.zeros(max(label_sf.keys()) + 1)
    for label_id in label_sf:
        label_sf_var[label_id] = np.var(label_sf[label_id])
    # print(sorted(label_sf_var))
    mean_var = np.mean(label_sf_var)
    return label_sf_var < var_threshold * mean_var

def gen_dis_mask(label_sf, dis_threshold):
    label_sf_dis = np.zeros(max(label_sf.keys()) + 1)
    for label_id in label_sf:
        label_sf_dis[label_id] = np.linalg.norm(np.mean(label_sf[label_id]))
    # print(sorted(label_sf_dis))
    mean_norm = np.mean(label_sf_dis)
    return label_sf_dis > dis_threshold * mean_norm

def run(cfg_file = 'gen_mos.yaml'):
    print('running...')
    with open(cfg_file) as file:
        cfg = yaml.safe_load(file)
    p1_id = cfg['p1_id']
    p2_id = cfg['p2_id']
    pc_path = cfg['pc_path']

    poses = load_poses(cfg['poses_path'])
    calib = load_calib(cfg['calib_path'])
    # pc1 = (poses[p1_id] @ calib @ pc1.T).T
    # pc2 = (np.linalg.inv(poses[p1_id] @ calib) @ poses[p2_id] @ calib @ pc2.T).T
    pc1 = load_vertex(pc_path + f'scans_{p1_id}.bin')[:, :3]
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd1.estimate_normals()
    pc1_normals = np.array(pcd1.normals)
    pc1 = pre_process(pc1, pc1_normals)
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    labels = cluster(pcd1)
    mask = np.zeros(len(pc1))
    for idx in p2_id:
        pc2 = load_vertex(pc_path + f'scans_{idx}.bin')[:, :3]
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc2)
        pcd2.estimate_normals()
        pc2_normals = np.array(pcd2.normals)
        pc2 = pre_process(pc2, pc2_normals)
        sf = flow_infer(pc1, pc2)
        label_sf, label_pt = gather_pt(sf, labels, pc1)

        mask_dis = gen_dis_mask(label_sf, cfg['dis_threshold'])
        mask_horizon = gen_horizon_mos_mask(label_sf, cfg['horizon_threshold'])
        mask_var = gen_var_mask(label_sf, cfg['var_threshold'])

        for i in range(len(pc1)):
            if(labels[i] >=0
               and mask_dis[labels[i]] == 1
               and mask_var[labels[i]] == 1
               and mask_horizon[labels[i]] == 1
            ):
                mask[i] += 1

    tot = len(p2_id)
    colors = []
    for i in range(len(pc1)):
        if(mask[i] > 0):
            colors.append([1, 0, 0])
        else:
            colors.append([0, 0, 1])

    pcd1.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd1])

if __name__ == '__main__':
    run()