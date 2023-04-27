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
    cur_pc1 = pc1_in[seg1]
    cur_pc2 = pc2_in[seg2]

    pc1 = torch.tensor(cur_pc1).to(torch.float).unsqueeze(0).to(device)
    pc2 = torch.tensor(cur_pc2).to(torch.float).unsqueeze(0).to(device)
    with torch.no_grad():
        cur_sf = model(pc1,pc2,pc1,pc2,5)
    res_sf[seg1] = cur_sf[-1].cpu().detach().numpy()
    return res_sf

def gather_sf(sf, labels):
    label_sf = {}
    for i in range(len(labels)):
        if labels[i] not in label_sf:
            label_sf[labels[i]] = [sf[i]]
        else:
            label_sf[labels[i]].append(sf[i])
    return label_sf


def cluster(pc, pc_normals):

    is_not_ground = (pc[:, 2] > -1.5)
    horizontal_normals = (np.abs(pc_normals[:, -1]) > .85) & (pc[:, 2] < -1.0)
    is_near = (np.amax(np.abs(pc), axis=1) < 35)
    mask = is_not_ground & (~horizontal_normals) & is_near
    ori_idx = np.arange(len(pc))[mask]
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
    return ins, labels_ids, ori_idx


def PCA(data):
    H = np.dot(data.T,data)
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)
    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

def gen_pca_mask(cluster_ids, sf, theta_threshold = 60, dis_threshold = 1.0):
    mask_pca = np.zeros(sf.shape[0])
    for label_id in cluster_ids:
        if(len(cluster_ids[label_id]) < 20): continue
        w, v = PCA(sf[np.array(cluster_ids[label_id])])
        v1 = v[:, 0]

        for i in cluster_ids[label_id]:
            cur_dis = np.linalg.norm(sf[i])
            cos_theta = np.dot(v1, sf[i]) / (np.linalg.norm(v1) * cur_dis)
            mask_pca[i] = (np.abs(cos_theta) < theta_threshold / 180 * np.pi) & (cur_dis > dis_threshold)
        # print(np.sum(mask_pca[cluster_ids[label_id]]) / len(cluster_ids[label_id]))
    return mask_pca

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
            extent = bbox.extent
            axis_lengths = [2 * extent[i] for i in range(3)]
            if all([axis_lengths[i] < 15.0 for i in range(3)]):
                box[i] = np.asarray(bbox.get_box_points())
    return box

def open_label(filename):
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))

    sem_label = label & 0xFFFF
    label = [1 if i > 250 else 0 for i in sem_label ]
    return label

def open_mos(filename, size):
    mos_id = np.load(filename)
    label = np.zeros(size)
    for it in mos_id:
        label[it] = 1
    return label

def run(cfg_file = 'cfg/box.yaml'):
    print('running...')
    with open(cfg_file) as file:
        cfg = yaml.safe_load(file)
    poses = load_poses(cfg['poses_path'])
    calib = load_calib(cfg['calib_path'])
    pc_path = cfg['pc_path']
    seq_acc = []
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    for p1_id in range(9, 4071, 10):
        p2_id = [p1_id - cfg['skip_n']]
        pchom1 = load_vertex(pc_path + f'{p1_id:06d}.bin')
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pchom1[:, :3])
        pcd1.estimate_normals()
        pc1_normals = np.array(pcd1.normals)
        ins1, cluster_ids1, ori_idx1 = cluster(pchom1[:, :3], pc1_normals)
        box1 = gen_box(cluster_ids1, pchom1[:, :3])

        mask = np.zeros(len(pchom1))
        for idx in p2_id:
            pchom2 = load_vertex(pc_path + f'{idx:06d}.bin')
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(pchom2[:, :3])
            pcd2.estimate_normals()
            pc2_normals = np.array(pcd2.normals)
            pchom2 = (np.linalg.inv(poses[p1_id] @ calib) @ poses[idx] @ calib @ pchom2.T).T
            _, cluster_ids2, ori_idx2 = cluster(pchom2[:, :3], pc2_normals)
            box2 = gen_box(cluster_ids2, pchom2[:, :3])
            mask_iou = {}
            for i in box1:
                mask_iou[i] = 1
                for j in box2:
                    if iou(box1[i], box2[j]) > cfg['iou_threshold']:
                        mask_iou[i] = 0

            sf = flow_infer(pchom1[:, :3], pchom2[:, :3], cfg['checkpoint'], ori_idx1, ori_idx2)
            
            mask_pca = gen_pca_mask(cluster_ids1, sf, cfg['theta_threshold'],cfg['dis_threshold'])
            for i in range(len(mask_pca)):
                if(mask_pca[i] == 1 and 
                (i in ins1) and 
                (ins1[i] in mask_iou) and 
                mask_iou[ins1[i]] == 1):
                    mask[i] += 1

        tot = len(p2_id)

        is_mos = np.zeros(len(pchom1))
        for i in cluster_ids1:
            if (len(cluster_ids1[i]) >= cfg['num_threshold'] and 
                np.sum(mask[cluster_ids1[i]]) / (tot * len(cluster_ids1[i])) > cfg['check_ratio']):
                is_mos[cluster_ids1[i]] = 1

        gt_label = open_mos(cfg['label_path'] + f'{p1_id:06d}.npy', len(pchom1))
        tp, fp, fn = 0, 0, 0
        for i in range(len(is_mos)):
            if(is_mos[i] == 1 and gt_label[i] == 1):
                tp+=1
            elif (is_mos[i] == 1 and gt_label[i] == 0):
                fp+=1
            elif (is_mos[i] == 0 and gt_label[i] == 1):
                fn+=1
        tot_tp += tp
        tot_fn += fn
        tot_fp += fp
        if tp + fp + fn == 0: continue
        seq_acc.append(tp / (tp + fp + fn))

        print(p1_id, sum(seq_acc) / len(seq_acc), tot_tp/(tot_tp+tot_fn+tot_fp))

if __name__ == '__main__':
    run()