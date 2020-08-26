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
    DATA_DIR = osp.join(ROOT_DIR, "model_logs/stage3_root2/result/stage3_root2_generate_train_test_.json")
    
    OUTPUT_DIR = osp.join(ROOT_DIR, 'model_logs',
            osp.split(osp.split(osp.realpath(__file__))[0])[1])

    # -------- Data Config -------- #
    DATALOADER = edict()
    DATALOADER.NUM_WORKERS = 0
    DATALOADER.ASPECT_RATIO_GROUPING = False
    DATALOADER.SIZE_DIVISIBILITY = 0

    DATASET = edict()
    DATASET.ROOT_IDX = 2  # pelvis
    DATASET.MAX_PEOPLE = 20

    # -------- Model Config -------- #
    MODEL = edict()
    MODEL.DEVICE = 'cuda'
    MODEL.GPU_IDS = [0]

    # -------- Training Config -------- #
    SOLVER = edict()
    SOLVER.BASE_LR = 0.08
    SOLVER.BATCH_SIZE = 1024
    SOLVER.NUM_EPOCHS = 200
    SOLVER.LR_STEP_SIZE = 30
    SOLVER.GAMMA = 0.5

    # --------- Checkpoint Config -------- #
    PRINT_FREQ = 1
    SAVE_FREQ = 1
    CHECKPOINT_DIR = osp.join(OUTPUT_DIR, "moco.root_2")

    # --------- Testing Config ----------- #
    TEST = edict()
    TEST.BATCH_SIZE = 32


cfg = Config()

def link_log_dir():
    if not osp.exists('./log'):
        ensure_dir(cfg.OUTPUT_DIR)
        cmd = 'ln -s ' + cfg.OUTPUT_DIR + ' log'
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
