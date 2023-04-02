import open3d as o3d
import numpy as np
import torch
from experiment import SceneFlowExp
from models import *
import yaml
device = torch.device('cuda')
print(device)
cpu = torch.device('cpu')
pc1 = torch.tensor(np.load('/media/dl/data_pc/data_demo_pre/KITTI_processed_occ_final/000000/pc1.npy')).unsqueeze(0).to(device)
pc2 = torch.tensor(np.load('/media/dl/data_pc/data_demo_pre/KITTI_processed_occ_final/000000/pc2.npy')).unsqueeze(0).to(device)

config = yaml.safe_load(open('configs/test/flowstep3d_sv.yaml'))
model = models_dict[config['model_params']['model_name']](**config['model_params'])
experiment = SceneFlowExp(model, config['exp_params'])
checkpoint = torch.load('/media/dl/data_pc/ckpt/flowstep3d_checkpoints/flowstep3d_sv/2020-10-19_13-17/epoch=33.ckpt')
experiment.load_state_dict(checkpoint['state_dict'])
experiment.eval().to(device)
with torch.no_grad():
    sf = experiment(pc1,pc2,pc1,pc2,5)

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

alpha = 1.2
mask_mos_idx = pdis > alpha * mean_norm

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pc2[:,:3])
# pcd2.paint_uniform_color([0,0,1])

colors = np.array([[1, 0, 0] if i == 1 else [0, 0, 1] for i in mask_mos_idx]) # 动态的是红色
pcd2.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd2])