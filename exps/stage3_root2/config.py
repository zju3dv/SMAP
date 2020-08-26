# encoding: utf-8
import os, getpass
import os.path as osp
import argparse

from easydict import EasyDict as edict
from dataset.data_settings import load_dataset
from cvpack.utils.pyt_utils import ensure_dir


class Config:
    # -------- Directoy Config -------- #
    ROOT_DIR = os.environ['PROJECT_HOME']
    OUTPUT_DIR = osp.join(ROOT_DIR, 'model_logs', osp.split(osp.split(osp.realpath(__file__))[0])[1])
    TEST_DIR = osp.join(OUTPUT_DIR, 'log_dir')
    TENSORBOARD_DIR = osp.join(OUTPUT_DIR, 'tb_dir') 

    # -------- Data Config -------- #
    DATALOADER = edict()
    DATALOADER.NUM_WORKERS = 0
    DATALOADER.ASPECT_RATIO_GROUPING = False
    DATALOADER.SIZE_DIVISIBILITY = 0

    DATASET = edict()
    DATASET.NAME = 'MIX'
    dataset = load_dataset(DATASET.NAME)
    DATASET.KEYPOINT = dataset.KEYPOINT
    DATASET.PAF = dataset.PAF
    DATASET.ROOT_IDX = dataset.ROOT_IDX  # pelvis or neck
    DATASET.MAX_PEOPLE = 20

    INPUT = edict()
    INPUT.NORMALIZE = True
    INPUT.MEANS = [0.406, 0.456, 0.485]  # bgr
    INPUT.STDS = [0.225, 0.224, 0.229]
    INPUT_SHAPE = dataset.INPUT_SHAPE
    OUTPUT_SHAPE = dataset.OUTPUT_SHAPE

    # -------- Model Config -------- #
    MODEL = edict()
    MODEL.STAGE_NUM = 3
    MODEL.UPSAMPLE_CHANNEL_NUM = 256

    MODEL.DEVICE = 'cuda'

    MODEL.WEIGHT = None  # osp.join(ROOT_DIR, 'lib/models/resnet-50_rename.pth')

    # -------- Training Config -------- #
    SOLVER = edict()
    SOLVER.IMG_PER_GPU = 2
    SOLVER.BASE_LR = 2e-4
    SOLVER.CHECKPOINT_PERIOD = 4800
    SOLVER.MAX_ITER = 96000*2
    SOLVER.WEIGHT_DECAY = 8e-6
    SOLVER.WARMUP_FACTOR = 0.1
    SOLVER.WARMUP_ITERS = 2400 

    LOSS = edict()
    LOSS.OHKM = True
    LOSS.TOPK = 8
    LOSS.COARSE_TO_FINE = True

    WITH_MDS = True
    RUN_EFFICIENT = False 
    
    # -------- Test Config -------- #
    TEST = edict()
    TEST.IMG_PER_GPU = 16
    TEST.ROOT_PATH = '/data/MultiPersonTestSet'  # '/data/datasets/mupots-3d-eval/MultiPersonTestSet'
    TEST.JSON_PATH = osp.join(TEST.ROOT_PATH, 'M3E_gt.json')


config = Config()
cfg = config


def link_log_dir():
    if not osp.exists('./log'):
        ensure_dir(config.OUTPUT_DIR)
        cmd = 'ln -s ' + config.OUTPUT_DIR + ' log'
        os.system(cmd)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-log', '--linklog', default=False, action='store_true')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if args.linklog:
        link_log_dir()
