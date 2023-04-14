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


def flow_infer(pc1_in, pc2_in,ckpt_path, seg1, seg2):
    checkpoint = torch.load(ckpt_path)
    model = FlowStep3D(**checkpoint["hyper_parameters"])
    model_weights = checkpoint["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval().to(device)
    res_sf = np.zeros((len(pc1_in), 3))
    for i in range(8):
        cur_idx_1 = np.arange(len(seg1))[seg1 == i]
        cur_pc1 = pc1_in[cur_idx_1]
        cur_pc2 = pc2_in[seg2 == i]

        if (len(cur_pc1) < 100): continue
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(cur_pc1)
        labels1 = np.asarray(pcd1.cluster_dbscan(eps=0.55, min_points=20))
        cur_pc1 = cur_pc1[labels1 >= 0]
        if (len(cur_pc1) < 100): continue

        if(len(cur_pc2) < 100): continue
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(cur_pc2)
        labels2 = np.asarray(pcd2.cluster_dbscan(eps=0.55, min_points=20))
        cur_pc2 = cur_pc2[labels2 >= 0]
        if(len(cur_pc2) < 100): continue

        print(len(cur_pc1))
        pc1 = torch.tensor(cur_pc1).to(torch.float).unsqueeze(0).to(device)
        pc2 = torch.tensor(cur_pc2).to(torch.float).unsqueeze(0).to(device)
        with torch.no_grad():
            cur_sf = model(pc1,pc2,pc1,pc2,5)
        res_sf[cur_idx_1[labels1 >= 0]] = cur_sf[-1].cpu().detach().numpy()
    return res_sf

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


def cluster(pc, seg):
    ori_idx = np.arange(len(pc))[seg < 8]
    pc = pc[ori_idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    labels = np.asarray(pcd.cluster_dbscan(eps=0.55, min_points=10))
    labels_ids = {}
    for i in range(len(labels)):
        if labels[i] < 0: continue
        if labels[i] not in labels_ids:
            labels_ids[labels[i]] = [i]
        else:
            labels_ids[labels[i]].append(ori_idx[i])
    return labels_ids


def PCA(data):
    H = np.dot(data.T,data)
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)
    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors

def gen_pca_mask(cluster_ids, sf, theta_threshold = 60, dis_threshold = 0.5):
    mask_pca = np.ones(sf.shape[0])
    for label_id in cluster_ids:
        if(len(cluster_ids[label_id]) < 50): continue
        w, v = PCA(sf[np.array(cluster_ids[label_id])])
        v1 = v[:, 0]

        # mos pca
        for i in cluster_ids[label_id]:
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

def open_label(filename):
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))

    sem_label = label & 0xFFFF
    label = [1 if i > 250 else 0 for i in sem_label ]
    return label

def run(cfg_file = 'cfg/flowsegv2trans.yaml'):
    print('running...')
    with open(cfg_file) as file:
        cfg = yaml.safe_load(file)
    poses = load_poses(cfg['poses_path'])
    calib = load_calib(cfg['calib_path'])
    pc_path = cfg['pc_path']
    seq_acc = []
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    for p1_id in range(9, 4071):
        seg1 = np.load(cfg['seg_label'] + f'{p1_id:06d}.npy')

        p2_id = [p1_id - cfg['skip_n']]
        pchom1 = load_vertex(pc_path + f'{p1_id:06d}.bin')

        cluster_ids = cluster(pchom1[:, :3], seg1)
        mask = np.zeros(len(pchom1))
        for idx in p2_id:
            pchom2 = load_vertex(pc_path + f'{idx:06d}.bin')
            seg2 = np.load(cfg['seg_label'] + f'{idx:06d}.npy')
            pchom2 = (np.linalg.inv(poses[p1_id] @ calib) @ poses[idx] @ calib @ pchom2.T).T
            sf = flow_infer(pchom1[:, :3], pchom2[:, :3], cfg['checkpoint'], seg1, seg2)

            mask_pca = gen_pca_mask(cluster_ids, sf, cfg['theta_threshold'],cfg['dis_threshold'])

            for i in range(len(mask_pca)):
                if(mask_pca[i] == 1):
                    mask[i] += 1

        tot = len(p2_id)

        is_mos = np.zeros(len(pchom1))
        for i in cluster_ids:
            if len(cluster_ids[i]) >= cfg['num_threshold'] and np.sum(mask[cluster_ids[i]]) / (tot * len(cluster_ids[i])) > cfg['check_ratio']:
                is_mos[cluster_ids[i]] = 1

        gt_label = open_label(cfg['label_path'] + f'{p1_id:06d}.label' )
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
        np.save(cfg['save_label'] + f'{p1_id:06d}', is_mos)

    print(sum(seq_acc) / len(seq_acc))

if __name__ == '__main__':
    run()