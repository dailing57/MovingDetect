import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import open3d as o3d
import numpy as np
import torch
import math
from models.flowstep3d import FlowStep3D
import yaml
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda')

np.random.seed(57)
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
    is_not_ground = (pc[:, 2] > -1.5)
    horizontal_normals = (np.abs(pc_normals[:, -1]) > .85) & (pc[:, 2] < -1.0)
    is_near = (np.amax(np.abs(pc), axis=1) < 35)
    mask = is_not_ground & (~horizontal_normals) & is_near
    ori_idx = np.arange(len(pc))[mask]

    pc = pc[ori_idx]

    return pc, ori_idx

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

    checkpoint = torch.load('/media/dl/data_pc/ckpt/flowstep3d_finetune/2023-04-05_20-02/epoch=9.ckpt')
    model = FlowStep3D(**checkpoint["hyper_parameters"])
    model_weights = checkpoint["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    model.load_state_dict(model_weights)

    model.eval().to(device)
    with torch.no_grad():
        sf = model(pc1,pc2,pc1,pc2,10)

    return sf[-1].cpu().detach().numpy().reshape(-1, 3)

def gather_sf(sf, labels):
    label_sf = {}
    for i in range(len(labels)):
        if labels[i] not in label_sf:
            label_sf[labels[i]] = [sf[i]]
        else:
            label_sf[labels[i]].append(sf[i])
    return label_sf

def gather_label(labels, pc1):
    label_pt, label_pt_id = {}, {}
    for i in range(len(labels)):
        if labels[i] not in label_pt:
            label_pt[labels[i]] = [pc1[i]]
            label_pt_id[labels[i]] = [i]
        else:
            label_pt[labels[i]].append(pc1[i])
            label_pt_id[labels[i]]. append(i)
    return label_pt, label_pt_id

def PCA(data):
    H = np.dot(data.T,data)
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)
    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

def gen_hist(distance_list, title = ''):
    plt.hist(distance_list, bins=100)
    plt.title(title)
    plt.show()

def gen_pca_mask(label_sf, label_pt_id, sf, theta_threshold = 60, dis_threshold = 0.5):
    mask_pca = np.ones(sf.shape[0])
    for label_id in label_sf:
        if label_id < 0: continue
        w, v = PCA(np.array(label_sf[label_id]))
        v1 = v[:, 0]

        for i in label_pt_id[label_id]:
            cur_dis = np.linalg.norm(sf[i])
            cos_theta = np.dot(v1, sf[i]) / (np.linalg.norm(v1) * cur_dis)
            mask_pca[i] = (np.abs(cos_theta) < theta_threshold / 180 * np.pi) & (cur_dis > dis_threshold)
    return mask_pca


def gen_var_mask(label_sf, var_threshold):
    label_sf_var = np.zeros(max(label_sf.keys()) + 1)
    for label_id in label_sf:
        if label_id < 0: continue
        label_sf_var[label_id] = np.sum(np.var(label_sf[label_id], axis=1))
    # gen_hist(label_sf_var, 'var')
    mean_var = np.mean(label_sf_var)
    print(mean_var)
    return label_sf_var < var_threshold * mean_var

def gen_dis_mask(label_sf, dis_threshold):
    label_sf_dis = np.zeros(max(label_sf.keys()) + 1)
    for label_id in label_sf:
        if label_id < 0: continue
        label_sf_dis[label_id] = np.linalg.norm(np.mean(label_sf[label_id], axis=1))
    # gen_hist(label_sf_dis, 'dis')
    mean_norm = np.mean(label_sf_dis)
    print(mean_norm)
    return label_sf_dis > dis_threshold * mean_norm

def run(cfg_file = 'cfg/flowsegv2.yaml'):
    print('running...')
    with open(cfg_file) as file:
        cfg = yaml.safe_load(file)
    p1_id = cfg['p1_id']
    p2_id = cfg['p2_id']
    pc_path = cfg['pc_path']

    poses = load_poses(cfg['poses_path'])
    calib = load_calib(cfg['calib_path'])
    # pc1 = (poses[p1_id] @ calib @ pc1.T).T
    pchom1 = load_vertex(pc_path + f'{p1_id:06d}.bin')
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pchom1[:, :3])
    pcd1.estimate_normals()
    pc1_normals = np.array(pcd1.normals)
    pc1, ori_idx = pre_process(pchom1[:, :3], pc1_normals)
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    labels = cluster(pcd1)
    label_pt, label_pt_id = gather_label(labels, pc1)
    mask = np.zeros(len(pc1))
    for idx in p2_id:
        pchom2 = load_vertex(pc_path + f'{idx:06d}.bin')
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pchom2[:, :3])
        pcd2.estimate_normals()
        pc2_normals = np.array(pcd2.normals)
        pc2 = (np.linalg.inv(poses[p1_id] @ calib) @ poses[idx] @ calib @ pchom2.T).T
        pc2, _ = pre_process(pc2[:, :3], pc2_normals)
        sf = flow_infer(pc1, pc2)
        # gen_hist(np.linalg.norm(sf, axis=1))
        label_sf = gather_sf(sf, labels)

        mask_pca = gen_pca_mask(label_sf, label_pt_id, sf, cfg['theta_threshold'],cfg['dis_threshold'])

        for i in range(len(labels)):
            if(labels[i] >=0 and mask_pca[i] == 1):
                mask[i] += 1

    tot = len(p2_id)

    is_mos_cur = np.zeros(len(pc1))
    for i in label_pt_id:
        if i < 0: continue
        if len(label_pt_id[i]) >= 30 and np.sum(mask[label_pt_id[i]]) / (tot * len(label_pt_id[i])) > 0.9:
            is_mos_cur[label_pt_id[i]] = 1

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pchom1[:, :3])
    cloud_tree = o3d.geometry.KDTreeFlann(pcd1)
    is_mos_ori = np.zeros(len(pchom1))
    colors = np.array([[0.25, 0.25, 0.25] for i in range(len(pchom1))])
    for i in range(len(ori_idx)):
        if(is_mos_cur[i] == 1):
            is_mos_ori[ori_idx[i]] = 1
            colors[ori_idx[i]] = [1, 0, 0]
            [_, idx, _] = cloud_tree.search_radius_vector_3d(pcd1.points[ori_idx[i]], 0.2)
            is_mos_ori[idx] = 1
            colors[idx] = [1, 0, 0]



    pcd1.colors = o3d.utility.Vector3dVector(np.array(colors))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd1)
    # width, height, focal = 800, 600, 0.96
    # K = [[focal * width, 0, width / 2 - 0.5],
    #     [0, focal * width, height / 2 -0.5],
    #     [0, 0, 1]]
    # camera = o3d.camera.PinholeCameraParameters()
    # camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2])
    # camera.extrinsic = np.array( [[-9.40456774e-01,  3.32088603e-01,  7.25135505e-02, -8.86299589e-01],
    #                               [-5.55180191e-03,  1.98294206e-01, -9.80126821e-01,  1.43469784e+01],
    #                               [-3.39867964e-01, -9.22169490e-01, -1.84643440e-01,  3.88750768e+01],
    #                               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],], dtype=np.float64)

    # ctr = vis.get_view_control()
    # ctr.convert_from_pinhole_camera_parameters(camera)

    vis.run()
    # params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # print("View Matrix:\n", params.extrinsic)
    # print("Projection Matrix:\n", params.intrinsic)
    # float_buffer = np.array(vis.capture_screen_float_buffer(True))

    vis.destroy_window()
    # image = Image.fromarray((float_buffer * 255).astype('uint8'))
    # image.save('image.png')

if __name__ == '__main__':
    run()