import sys, os
import os.path as osp
import numpy as np

import torch.utils.data as data

__all__ = ['semantic3d']


class semantic3d(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        args:
    """
    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 overfit_samples=None,
                 full=True):
        self.root = data_root
        self.train = train
        self.transform = transform
        self.num_points = num_points

        self.samples = self.make_dataset(full, overfit_samples)

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def make_dataset(self, full, overfit_samples):
        root = osp.realpath(osp.expanduser(self.root))
        train_seq = [0, 1, 2, 4]
        val_seq = 8
        useful_paths = []
        if (self.train and overfit_samples is None):
            for seq in train_seq:
                cur_poses = self.load_poses(os.path.join(root, f'{seq:02d}/poses.txt'))
                cur_calib = self.load_calib(os.path.join(root, f'{seq:02d}/calib.txt'))
                cur_path = os.path.join(root, f'{seq:02d}/velodyne')
                for folder, subfolders, files in os.walk(cur_path):
                    files = sorted(files)
                    for i in range(1, len(files), 5):
                        useful_paths.append([os.path.join(cur_path, files[i-1]), 
                                            os.path.join(cur_path, files[i]), 
                                            cur_poses[i-1], cur_poses[i], cur_calib])
        else:
            cur_poses = self.load_poses(os.path.join(root, f'{val_seq:02d}/poses.txt'))
            cur_calib = self.load_calib(os.path.join(root, f'{val_seq:02d}/calib.txt'))
            cur_path = os.path.join(root, f'{val_seq:02d}/velodyne')
            for folder, subfolders, files in os.walk(cur_path):
                files = sorted(files)
                for i in range(1, len(files), 100):
                    useful_paths.append([os.path.join(cur_path, files[i-1]), 
                                        os.path.join(cur_path, files[i]), 
                                        cur_poses[i-1] @ cur_calib, cur_poses[i] @ cur_calib])

        if overfit_samples is not None:
            res_paths = useful_paths[:overfit_samples]
        else:
            if not full:
                res_paths = useful_paths[::4]
            else:
                res_paths = useful_paths

        return res_paths
    
    def str_to_trans(self, poses):
        return np.vstack((np.fromstring(poses, dtype=np.float32, sep=' ').reshape(3, 4), [0, 0, 0, 1]))

    def load_poses(self, pose_path):
        poses = []
        with open(pose_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                poses.append(self.str_to_trans(line))
        return np.array(poses)


    def load_calib(self, calib_path):
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    return self.str_to_trans(line.replace('Tr:', ''))
                
    def load_vertex(self, scan_path):
        current_vertex = np.fromfile(scan_path, dtype=np.float32).reshape((-1, 4))
        current_vertex[:,3] = np.ones(current_vertex.shape[0])
        return current_vertex

    def pc_loader(self, cur):
        pc1 = self.load_vertex(cur[0])
        pc2 = self.load_vertex(cur[1])
        pc1 = (cur[2] @ pc1.T).T
        pc2 = (cur[3] @ pc2.T).T
        return pc1[:, :3].astype(np.float32), pc2[:, :3].astype(np.float32)
