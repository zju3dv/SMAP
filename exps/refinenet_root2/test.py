import torch
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import copy
import time

from dataset.p2p_dataset import P2PDataset
from model.refinenet import RefineNet
from lib.utils.model_serialization import load_state_dict
from config import cfg


def main(opt):
    load_epoch = opt.load_epoch
    test_dataset = P2PDataset(dataset_path=cfg.DATA_DIR, root_idx=cfg.DATASET.ROOT_IDX)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)
    model = RefineNet()
    model = model.cuda()

    min_root_error = 1000
    min_idx = 0
    while True:
        ckpt_file = os.path.join(cfg.CHECKPOINT_DIR, "RefineNet_epoch_%03d.pth" % load_epoch)
        if not os.path.exists(ckpt_file):
            print("No ckpt of epoch {}".format(load_epoch))
            print("Best real_error iter is {}, error is {}".format(min_idx, min_root_error))
            break
        load_state_dict(model, torch.load(ckpt_file))
        model.eval()

        count = 0
        root_error = 0
        time_total = 0.0
        for i, (inp, gt_t) in enumerate(test_loader):
            inp = inp.cuda()
            gt_t = gt_t
            with torch.no_grad():
                start_time = time.time()
                pred_t = model(inp)
                time_total += time.time() - start_time
                pred_t = pred_t.cpu()
                # loss = criterion(pred, gt)
                for j in range(len(pred_t)):
                    gt = copy.deepcopy(gt_t[j].numpy())
                    gt.resize((15, 3))
                    pred = copy.deepcopy(pred_t[j].numpy())
                    pred.resize((15, 3))
                    count += 1
                    root_error += np.linalg.norm(np.abs(pred - gt), axis=1)

        print_root_error = root_error/count
        mean_root_error = np.mean(print_root_error)
        print("Root error of epoch {} is {}, mean is {}".format(load_epoch, print_root_error, mean_root_error))
        if mean_root_error < min_root_error:
            min_root_error = mean_root_error
            min_idx = load_epoch
        load_epoch += cfg.SAVE_FREQ
        print("Time per inference is {}".format(time_total / len(test_loader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_epoch', type=int, default=1)
    opt = parser.parse_args()
    main(opt)
