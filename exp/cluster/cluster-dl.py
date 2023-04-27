import open3d as o3d
import numpy as np
import math

def are_same_plane(plane1, plane2, tolerance):
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    
    dot_product = a1*a2 + b1*b2 + c1*c2
    if abs(dot_product) < 0.85:
        return False
    
    dist = abs(d1 - d2) / math.sqrt(a1**2 + b1**2 + c1**2)
    
    if dist <= tolerance:
        return True
    else:
        return False

np.random.seed(57)
# 读取点云
data_path = '/media/dl/data_pc/semanticKITTI/sequences/08/velodyne/000098.bin'
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

pc = load_vertex(data_path)[:,:3]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)
pcd.estimate_normals()
pc1_normals = np.array(pcd.normals)
pc = pre_process(pc, pc1_normals)
pcd.points = o3d.utility.Vector3dVector(pc)

labels = np.asarray(pcd.cluster_dbscan(eps=0.55, min_points=10))


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
    cloud = pcd.select_by_index(indices)
    # 拟合平面
    [a, b, c, d], idx = cloud.segment_plane(0.3, 3, 10)
    planes_idx.append(indices[idx])
    planes_arg.append([a,b,c,d])

for i in range(len(planes_arg)):
    for j in range(i + 1, len(planes_arg)):
        if labels[planes_idx[j][0]] == labels[planes_idx[i][0]]: continue
        if(are_same_plane(planes_arg[i], planes_arg[j], 0.3)):
            labels[planes_idx[j]] = labels[planes_idx[i][0]]

color_map = [np.random.uniform(0.25, 1, 3) for _ in range(max(labels) + 1)]
colors = [color_map[labels[idx]] if labels[idx] >= 0 else [0, 0, 0] for idx in range(len(pc))]
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])

