import open3d as o3d
import numpy as np
import torch
# from experiment import SceneFlowExp
from models.flowstep3d import FlowStep3D
import yaml
device = torch.device('cuda')
print(device)
cpu = torch.device('cpu')

seq = 5
p1_id = 100
p2_id = 101

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

pc_path = [f'/media/dl/data_pc/semanticKITTI/sequences/{seq:02d}/velodyne/{p1_id:06d}.bin',
           f'/media/dl/data_pc/semanticKITTI/sequences/{seq:02d}/velodyne/{p2_id:06d}.bin']
poses_path = f'/media/dl/data_pc/semanticKITTI/sequences/{seq:02d}/poses.txt'
calib_path = f'/media/dl/data_pc/semanticKITTI/sequences/{seq:02d}/calib.txt'

poses = load_poses(poses_path)
calib = load_calib(calib_path)
pc1 = load_vertex(pc_path[0])
pc2 = load_vertex(pc_path[1])

# pc1 = (poses[p1_id] @ calib @ pc1.T).T
pc2 = (np.linalg.inv(poses[p1_id] @ calib) @ poses[p2_id] @ calib @ pc2.T).T

# pc1 = pc1[:, [0, 2, 1]]
# pc2 = pc2[:, [0, 2, 1]]

pc1 = pc1[pc1[:, 2] > -1.4]
pc2 = pc2[pc2[:, 2] > -1.4]

pc1 = torch.tensor(pc1[:,:3]).to(torch.float).unsqueeze(0).to(device)
pc2 = torch.tensor(pc2[:,:3]).to(torch.float).unsqueeze(0).to(device)

config = yaml.safe_load(open('configs/test/flowstep3d_sv.yaml'))
checkpoint = torch.load('/media/dl/data_pc/ckpt/flowstep3d_finetune/2023-03-25_23-12/epoch=18.ckpt')
model = FlowStep3D(**checkpoint["hyper_parameters"])
model_weights = checkpoint["state_dict"]
for key in list(model_weights):
    model_weights[key.replace("model.", "")] = model_weights.pop(key)
model.load_state_dict(model_weights)

model.eval().to(device)
with torch.no_grad():
    sf = model(pc1,pc2,pc1,pc2,5)

sf = sf[-1].cpu().detach().numpy().reshape(-1, 3)
pc1 = pc1.cpu().numpy().reshape(-1, 3)
pc2 = pc2.cpu().numpy().reshape(-1, 3)
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(pc1[:,:3])
pcd1.paint_uniform_color([1,0,0])

pdis = np.sqrt(np.sum(sf ** 2, axis=1))
mean_norm = np.mean(pdis)
print(f'mean norm: {mean_norm}')

pcdm = pc1 + sf
pcdsf = o3d.geometry.PointCloud()
pcdsf.points = o3d.utility.Vector3dVector(pcdm[:,:3])
pcdsf.paint_uniform_color([0,1,0])

alpha = 2
mask_mos_idx = pdis > alpha * mean_norm

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pc2[:,:3])
# pcd2.paint_uniform_color([0,0,1])

colors = np.array([[1, 0, 0] if i == 1 else [0, 0, 1] for i in mask_mos_idx]) # 动态的是红色
pcd2.colors = o3d.utility.Vector3dVector(colors)
pcd1.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd1])