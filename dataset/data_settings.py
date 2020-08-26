"""
@author: Jianan Zhen
@contact: jnzhen99@163.com
"""

from easydict import EasyDict as edict
import os
import os.path as osp


class MIX:
    NAME = 'MIX'
    KEYPOINT = edict()
    KEYPOINT.NUM = 15
    '''The order in this work:
        (0-'neck'  1-'head'  2-'pelvis'  
        3-'left_shoulder'  4-'left_elbow'  5-'left_wrist'
        6-'left_hip'  7-'left_knee'  8-'left_ankle'
        9-'right_shoulder'  10-'right_elbow'  11-'right_wrist'
        12-'right_hip'  13-'right_knee'  14-'right_ankle')
    '''
    KEYPOINT.FLIP_ORDER = [0, 1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8]
    ROOT_IDX = 2  # pelvis or neck

    PAF = edict()

    PAF.VECTOR = [[0, 1], [0, 2],
                  [0, 9], [9, 10], [10, 11],
                  [0, 3], [3, 4], [4, 5],
                  [2, 12], [12, 13], [13, 14],
                  [2, 6], [6, 7], [7, 8]]
    
    PAF.FLIP_CHANNEL = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 
                        22, 23, 24, 25, 26, 27, 16, 17, 18, 19, 20, 21]

    PAF.NUM = len(PAF.VECTOR)
    PAF.LINE_WIDTH_THRE = 1

    INPUT_SHAPE = (512, 832)  # height, width
    STRIDE = 4
    OUTPUT_SHAPE = (INPUT_SHAPE[0] // STRIDE, INPUT_SHAPE[1] // STRIDE)
    WIDTH_HEIGHT_RATIO = INPUT_SHAPE[1] / INPUT_SHAPE[0]
    
    PREFIX = os.environ['PROJECT_HOME']
    COCO_ROOT_PATH = osp.join(PREFIX, "data/coco2017")
    COCO_JSON_PATH = osp.join(COCO_ROOT_PATH, 'annotations/coco_keypoints_train2017.json')
    
    USED_3D_DATASETS = ["MUCO"]
    MUCO_ROOT_PATH = osp.join(PREFIX, "data/MuCo")
    MUCO_JSON_PATH = osp.join(MUCO_ROOT_PATH, "annotations/MuCo.json")
    CMUP_ROOT_PATH = osp.join(PREFIX, "data/Panoptic")
    CMUP_JSON_PATH = osp.join(CMUP_ROOT_PATH, 'annotations/Panoptic.json')
    H36M_ROOT_PATH = osp.join(PREFIX, "data/Human3.6M")
    H36M_JSON_PATH = osp.join(H36M_ROOT_PATH, "annotations/H36M.json")

    TRAIN = edict()
    TRAIN.CENTER_TRANS_MAX = 40
    TRAIN.ROTATE_MAX = 10
    TRAIN.FLIP_PROB = 0.5

    TRAIN.SCALE_MAX = 1.1
    TRAIN.SCALE_MIN = 0.8

    TRAIN.GAUSSIAN_KERNELS = [(15, 15), (11, 11), (9, 9), (7, 7), (5, 5)]


def load_dataset(name):
    if 'MIX' in name:
        dataset = MIX
        return dataset
    return None
