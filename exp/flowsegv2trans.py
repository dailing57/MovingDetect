import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import open3d as o3d
import numpy as np
import torch
from models.flowstep3d import FlowStep3D
import yaml
from shapely.geometry import MultiPoint


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


def flow_infer(pc1_in, pc2_in,ckpt_path, seg1, seg2):
    checkpoint = torch.load(ckpt_path)
    model = FlowStep3D(**checkpoint["hyper_parameters"])
    model_weights = checkpoint["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval().to(device)
    res_sf = np.zeros((len(pc1_in), 3))
    cur_idx_1 = np.arange(len(seg1))[(seg1 != 8) & (seg1 != 10)]
    cur_pc1 = pc1_in[cur_idx_1]
    cur_pc2 = pc2_in[(seg2 != 8) & (seg2 != 10)]

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(cur_pc1)
    labels1 = np.asarray(pcd1.cluster_dbscan(eps=0.55, min_points=20))
    cur_idx_1 = cur_idx_1[labels1 >= 0]
    cur_pc1 = cur_pc1[labels1 >= 0]

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(cur_pc2)
    labels2 = np.asarray(pcd2.cluster_dbscan(eps=0.55, min_points=20))
    cur_pc2 = cur_pc2[labels2 >= 0]

    print(len(cur_pc1))
    pc1 = torch.tensor(cur_pc1).to(torch.float).unsqueeze(0).to(device)
    pc2 = torch.tensor(cur_pc2).to(torch.float).unsqueeze(0).to(device)
    with torch.no_grad():
        cur_sf = model(pc1,pc2,pc1,pc2,5)
    res_sf[cur_idx_1] = cur_sf[-1].cpu().detach().numpy()
    return res_sf

def gather_sf(sf, labels):
    label_sf = {}
    for i in range(len(labels)):
        if labels[i] not in label_sf:
            label_sf[labels[i]] = [sf[i]]
        else:
            label_sf[labels[i]].append(sf[i])
    return label_sf

def cluster(pc, seg):
    ori_idx = np.arange(len(pc))[seg < 8]
    pc = pc[ori_idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    labels = np.asarray(pcd.cluster_dbscan(eps=0.55, min_points=10))
    labels_ids, ins = {}, {}
    for i in range(len(labels)):
        if labels[i] < 0: continue
        ins[ori_idx[i]] = labels[i]
        if labels[i] not in labels_ids:
            labels_ids[labels[i]] = [ori_idx[i]]
        else:
            labels_ids[labels[i]].append(ori_idx[i])
    return ins, labels_ids


def PCA(data):
    H = np.dot(data.T,data)
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)
    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

def gen_pca_mask(seg, cluster_ids, sf, theta_threshold = 60, dis_threshold_list = [1.0]):
    mask_pca = np.zeros(sf.shape[0])
    for label_id in cluster_ids:
        if(len(cluster_ids[label_id]) < 20): continue
        w, v = PCA(sf[np.array(cluster_ids[label_id])])
        v1 = v[:, 0]

        for i in cluster_ids[label_id]:
            dis_threshold = dis_threshold_list[seg[i] in [0, 3, 4]]
            cur_dis = np.linalg.norm(sf[i])
            cos_theta = np.dot(v1, sf[i]) / (np.linalg.norm(v1) * cur_dis)
            mask_pca[i] = (np.abs(cos_theta) < theta_threshold / 180 * np.pi) & (cur_dis > dis_threshold)
        # print(np.sum(mask_pca[cluster_ids[label_id]]) / len(cluster_ids[label_id]))
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

def iou(box1, box2):
    box1 = MultiPoint(box1[:, :2]).convex_hull
    box2 = MultiPoint(box2[:, :2]).convex_hull
    inter = box1.intersection(box2).area
    union = box1.area + box2.area - inter
    iou = inter / union
    return iou


def gen_box(labels, pc):
    box = {}
    for i in labels:
        if len(labels[i]) > 20:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pc[labels[i]]))
            bbox = pcd.get_oriented_bounding_box()
            box[i] = np.asarray(bbox.get_box_points())
    return box

def run(cfg_file = 'cfg/flowsegv2trans.yaml'):
    print('running...')
    with open(cfg_file) as file:
        cfg = yaml.safe_load(file)
    p1_id = cfg['p1_id']
    p2_id = cfg['p2_id']
    pc_path = cfg['pc_path']
    seg1 = np.load(cfg['seg_label'] + f'{p1_id:06d}.npy')
    poses = load_poses(cfg['poses_path'])
    calib = load_calib(cfg['calib_path'])
    pchom1 = load_vertex(pc_path + f'{p1_id:06d}.bin')

    labels1, cluster_ids1 = cluster(pchom1[:, :3], seg1)
    box1 = gen_box(cluster_ids1, pchom1[:, :3])

    mask = np.zeros(len(pchom1))
    for idx in p2_id:
        pchom2 = load_vertex(pc_path + f'{idx:06d}.bin')
        seg2 = np.load(cfg['seg_label'] + f'{idx:06d}.npy')
        pchom2 = (np.linalg.inv(poses[p1_id] @ calib) @ poses[idx] @ calib @ pchom2.T).T
        _, cluster_ids2 = cluster(pchom2[:, :3], seg2)
        box2 = gen_box(cluster_ids2, pchom2[:, :3])
        mask_iou = {}
        for i in box1:
            mask_iou[i] = 1
            for j in box2:
                if iou(box1[i], box2[j]) > cfg['iou_threshold'][seg1[cluster_ids1[i][0]] in [0, 3, 4]]:
                    mask_iou[i] = 0

        sf = flow_infer(pchom1[:, :3], pchom2[:, :3], cfg['checkpoint'], seg1, seg2)

        mask_pca = gen_pca_mask(seg1, cluster_ids1, sf, cfg['theta_threshold'],cfg['dis_threshold'])
        for i in range(len(mask_pca)):
            if(mask_pca[i] == 1 and 
               (i in labels1) and 
               (labels1[i] in mask_iou) and 
               mask_iou[labels1[i]] == 1):
                mask[i] += 1

    tot = len(p2_id)

    is_mos = np.zeros(len(pchom1))
    for i in cluster_ids1:
        if (len(cluster_ids1[i]) >= cfg['num_threshold'][seg1[cluster_ids1[i][0]] in [0, 3, 4]] and 
            np.sum(mask[cluster_ids1[i]]) / (tot * len(cluster_ids1[i])) > cfg['check_ratio']):
            is_mos[cluster_ids1[i]] = 1


    colors = np.array([[0.25, 0.25, 0.25] for i in range(len(pchom1))])
    for i in range(len(is_mos)):
        if(is_mos[i] == 1):
            colors[i] = [1, 0, 0]

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pchom1[:, :3])
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