import open3d as o3d
import numpy as np

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
    label = [1 if i > 250 else 0 for i in sem_label ]
    return label
def open_mos(filename, size):
    mos_id = np.load(filename)
    label = np.zeros(size)
    for it in mos_id:
        label[it] = 1
    return label

seq = "08"
width, height, focal = 1920, 1080, 0.96
K = [[focal * width, 0, width / 2 - 0.5],
    [0, focal * width, height / 2 -0.5],
    [0, 0, 1]]
camera = o3d.camera.PinholeCameraParameters()
camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2])
camera.extrinsic = np.array(  [[ 9.99208314e-01,  2.29207597e-02,  3.25174328e-02,  6.97716278e-01],
                                [-3.10441364e-02, -6.19408486e-02,  9.97596909e-01, -1.72749819e+01],
                                [ 2.48798364e-02, -9.97816601e-01, -6.11802557e-02,  1.85796824e+02],
                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],], dtype=np.float64)
for f_id in range(4009, 4070, 10):
    str_fid = "%06d"%(f_id)
    print(str_fid)
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    scan_path = f'/media/dl/data_pc/semanticKITTI/sequences/{seq}/velodyne/{str_fid}.bin'
    label_path = f'/media/dl/data_pc/semanticKITTI/sequences/{seq}/labels/{str_fid}.label'
    poses = load_poses('/media/dl/data_pc/semanticKITTI/sequences/08/poses.txt')
    calib = load_calib('/media/dl/data_pc/semanticKITTI/sequences/08/calib.txt')
    pc = load_vertex(scan_path)
    pc[:, [1, 2]] = pc[:, [2, 1]]
    pc = (poses[f_id].dot(pc.T)).T
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc[:, :3]))
    label = open_mos(f'/media/dl/data_pc/data_demo/mos_label/08/{f_id:06d}.npy', len(pc))
    colors = []
    for i in range(len(label)):
        if label[i] == 1:
            colors.append([1, 0, 0])
        else:
            colors.append([0.25,0.25,0.25])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()



