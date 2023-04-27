import open3d as o3d
import numpy as np
from shapely.geometry import MultiPoint

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
            
def open_label(filename):
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))

    sem_label = label & 0xFFFF
    ins_label = label >> 16
    # label = [1 if i > 250 else 0 for i in sem_label ]
    return sem_label, ins_label

def iou(box1, box2):
    box1 = MultiPoint(box1[:, :2]).convex_hull
    box2 = MultiPoint(box2[:, :2]).convex_hull
    inter = box1.intersection(box2).area
    union = box1.area + box2.area - inter
    iou = inter / union

    return iou



def gen_box(labels, pc):
    label_pt, label_pt_id, label_rec, bbox = {}, {}, {}, {}
    for i in range(len(labels)):
        if labels[i] < 0: continue
        if labels[i] not in label_pt:
            label_pt[labels[i]] = [pc[i]]
            label_pt_id[labels[i]] = [i]
        else:
            label_pt[labels[i]].append(pc[i])
            label_pt_id[labels[i]]. append(i)
    for i in label_pt:
        if len(label_pt[i]) > 20:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(label_pt[i]))
            bbox[i] = pcd.get_oriented_bounding_box()
            label_rec[i] = np.asarray(bbox[i].get_box_points())
    
    return label_pt_id, label_rec, bbox

seq = "08"
iou_thresh = [0.3, 0.5]
poses = load_poses('/media/dl/data_pc/semanticKITTI/sequences/08/poses.txt')
calib = load_calib('/media/dl/data_pc/semanticKITTI/sequences/08/calib.txt')
for f_id in range(109, 150, 10):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    scan_path = [f'/media/dl/data_pc/semanticKITTI/sequences/{seq}/velodyne/{f_id:06d}.bin',
                 f'/media/dl/data_pc/semanticKITTI/sequences/{seq}/velodyne/{(f_id - 2):06d}.bin',]
    label_path = [f'/media/dl/data_pc/semanticKITTI/sequences/{seq}/labels/{f_id:06d}.label',
                  f'/media/dl/data_pc/semanticKITTI/sequences/{seq}/labels/{(f_id - 2):06d}.label',]
    pc1 = load_vertex(scan_path[0])
    # pc1[:, [1, 2]] = pc1[:, [2, 1]]
    # poses[f_id - 2][3,[1, 2]] = poses[f_id - 2][3,[2, 1]]
    # poses[f_id][3,[1, 2]] = poses[f_id][3,[2, 1]]
    pc1 = (np.linalg.inv(poses[f_id - 2] @ calib) @ poses[f_id] @ calib @ pc1.T).T
    # pc1[:, [1, 2]] = pc1[:, [2, 1]]
    sem1, ins1 = open_label(label_path[0])
    pc2 = load_vertex(scan_path[1])
    sem2, ins2 = open_label(label_path[1])

    pc1 = pc1[:, :3]
    pc2 = pc2[:, :3]
    for i in range(252, 260):
        cur_idx1 = np.arange(len(pc1))[sem1==i]
        if(len(cur_idx1) < 20): continue
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1[cur_idx1])
        labels1 = np.asarray(pcd1.cluster_dbscan(eps=0.55, min_points=20))
        idx1, box1, bbox1 = gen_box(labels1, pc1[cur_idx1])

        cur_idx2 = np.arange(len(pc2))[sem2==i]
        if(len(cur_idx2) < 20): continue
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc2[cur_idx2])
        # labels2 = np.asarray(pcd2.cluster_dbscan(eps=0.55, min_points=20))
        idx2, box2, bbox2 = gen_box(ins2[cur_idx2], pc2[cur_idx2])

        for i in box1:
            for j in box2:
                if iou(box1[i], box2[j]) > iou_thresh[sem1[cur_idx1[idx1[i][0]]] in [252, 256, 256, 257, 258, 259]]:
                    bbox1[i].color = (1,0,0)
                    bbox2[j].color = (0,0,1)
                    vis.add_geometry(bbox1[i])
                    vis.add_geometry(bbox2[j])
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc1))
    colors = []
    for i in range(len(sem1)):
        if sem1[i] > 250:
            colors.append([1, 0, 0])
        else:
            colors.append([0.25,0.25,0.25])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(pc2)
    # pcd2.paint_uniform_color([0,0,1])
    # vis.add_geometry(pcd2)
    vis.run()
    vis.destroy_window()



