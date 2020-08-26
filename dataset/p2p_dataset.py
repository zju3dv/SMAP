import json
import os.path as osp

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class P2PDataset(Dataset):
    def __init__(self, stage='train', dataset_path='', root_idx=2):
        self.root_idx = root_idx
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)['3d_pairs']

    def __getitem__(self, index):
        pair = self.dataset[index]
        input_point_3d = np.asarray(pair['pred_3d'], dtype=np.float)
        input_point_2d = np.asarray(pair['pred_2d'], dtype=np.float)
        gt_point_3d = np.asarray(pair['gt_3d'], dtype=np.float)

        input_point = np.zeros((15, 5), dtype=np.float)  
        gt_point = np.zeros((15, 3), dtype=np.float)    
        gt_point[self.root_idx] = 0  # relative to the root joint
        input_point[self.root_idx, :2] = input_point_2d[self.root_idx, :2]
        input_point[self.root_idx, 2:] = input_point_3d[self.root_idx, :3]
        for i in range(0, len(input_point_2d)):
            if i != self.root_idx:
                gt_point[i] = gt_point_3d[i] - gt_point_3d[self.root_idx]
                if input_point_3d[i, 3] > 0:
                    input_point[i, :2] = input_point_2d[i, :2] - input_point_2d[self.root_idx, :2]
                    input_point[i, 2:] = input_point_3d[i, :3] - input_point_3d[self.root_idx, :3]
                
        inp = input_point.flatten()
        gt = gt_point.flatten()
        inp = torch.from_numpy(inp).float()
        gt = torch.from_numpy(gt).float()
        return inp, gt

    def __len__(self):
        return len(self.dataset)
