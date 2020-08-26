import os.path as osp
import glob
import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, cfg, dataset_path):
        
        self.dataset_path = dataset_path

        self.image_list = glob.glob(osp.join(dataset_path, '**/*.jpg'), recursive=True)
        self.image_list.extend(glob.glob(osp.join(dataset_path, '**/*.png'), recursive=True))
        self.image_list.extend(glob.glob(osp.join(dataset_path, '**/*.jpeg'), recursive=True))
        self.list_size = len(self.image_list)
        
        self.net_input_shape = (832, 512) # (width, height)

        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_list[index].rstrip()
        image_name = image_path.replace(self.dataset_path, '').lstrip('/')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.image_shape = (image.shape[1], image.shape[0])  # --> (width, heght)

        net_input_image, scale = self.aug_croppad(image)
        net_input_image = self.transform(net_input_image)

        return net_input_image, image_name, scale

    def __len__(self):
        return self.list_size

    def aug_croppad(self, img):
        scale = dict()
        crop_x = self.net_input_shape[0]  # width
        crop_y = self.net_input_shape[1]  # height
        scale['scale'] = min(crop_x / self.image_shape[0], crop_y / self.image_shape[1])
        img = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = self.image_shape[0]
        scale['img_height'] = self.image_shape[1]
        scale['net_width'] = self.net_input_shape[0]
        scale['net_height'] = self.net_input_shape[1]

        center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int)
        
        if img.shape[1] < crop_x:    # pad left and right
            margin_l = (crop_x - img.shape[1]) // 2
            margin_r = crop_x - img.shape[1] - margin_l
            pad_l = np.ones((img.shape[0], margin_l, 3), dtype=np.uint8) * 128
            pad_r = np.ones((img.shape[0], margin_r, 3), dtype=np.uint8) * 128
            img = np.concatenate((pad_l, img, pad_r), axis=1)
        elif img.shape[0] < crop_y:  # pad up and down
            margin_u = (crop_y - img.shape[0]) // 2
            margin_d = crop_y - img.shape[0] - margin_u
            pad_u = np.ones((margin_u, img.shape[1], 3), dtype=np.uint8) * 128
            pad_d = np.ones((margin_d, img.shape[1], 3), dtype=np.uint8) * 128
            img = np.concatenate((pad_u, img, pad_d), axis=0)
        
        return img, scale
