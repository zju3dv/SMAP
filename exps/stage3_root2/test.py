"""
@author: Jianan Zhen
@contact: zhenjianan@sensetime.com
"""

import os
import argparse
import json
import cv2
from tqdm import tqdm

from torch.utils.data import DataLoader

from cvpack.utils.logger import get_logger
from model.smap import SMAP
from model.refinenet import RefineNet
from lib.utils.dataloader import get_test_loader
from lib.utils.comm import is_main_process
from exps.stage3_root2.test_util import *
from dataset.custom_dataset import CustomDataset
from config import cfg
import dapalib


def generate_3d_point_pairs(model, refine_model, data_loader, cfg, logger, device, 
                            output_dir=''):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    if refine_model is not None:
        refine_model.eval()

    result = dict()
    result['model_pattern'] = cfg.DATASET.NAME
    result['3d_pairs'] = []
    # 3d_pairs has items like{'pred_2d':[[x,y,detZ,score]...], 'gt_2d':[[x,y,Z,visual_type]...],
    #                         'pred_3d':[[X,Y,Z,score]...], 'gt_3d':[[X,Y,X]...],
    #                         'root_d': (abs depth of root (float value) pred by network),
    #                         'image_path': relative image path}

    kpt_num = cfg.DATASET.KEYPOINT.NUM
    data = tqdm(data_loader) if is_main_process() else data_loader
    for idx, batch in enumerate(data):
        if cfg.TEST_MODE == 'run_inference':
            imgs, img_path, scales = batch
            meta_data = None
        else:
            imgs, meta_data, img_path, scales = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            outputs_2d, outputs_3d, outputs_rd = model(imgs)

            outputs_3d = outputs_3d.cpu()
            outputs_rd = outputs_rd.cpu()

            if cfg.DO_FLIP:
                imgs_flip = torch.flip(imgs, [-1])
                outputs_2d_flip, outputs_3d_flip, outputs_rd_flip = model(imgs_flip)
                outputs_2d_flip = torch.flip(outputs_2d_flip, dims=[-1])
                # outputs_3d_flip = torch.flip(outputs_3d_flip, dims=[-1])
                # outputs_rd_flip = torch.flip(outputs_rd_flip, dims=[-1])
                keypoint_pair = cfg.DATASET.KEYPOINT.FLIP_ORDER
                paf_pair = cfg.DATASET.PAF.FLIP_CHANNEL
                paf_abs_pair = [x+kpt_num for x in paf_pair]
                pair = keypoint_pair + paf_abs_pair
                for i in range(len(pair)):
                    if i >= kpt_num and (i - kpt_num) % 2 == 0:
                        outputs_2d[:, i] += outputs_2d_flip[:, pair[i]]*-1
                    else:
                        outputs_2d[:, i] += outputs_2d_flip[:, pair[i]]
                outputs_2d[:, kpt_num:] *= 0.5

            for i in range(len(imgs)):
                if meta_data is not None:
                    # remove person who was blocked
                    new_gt_bodys = []
                    annotation = meta_data[i].numpy()
                    scale = scales[i]
                    for j in range(len(annotation)):
                        if annotation[j, cfg.DATASET.ROOT_IDX, 3] > 1:
                            new_gt_bodys.append(annotation[j])
                    gt_bodys = np.asarray(new_gt_bodys)
                    if len(gt_bodys) == 0:
                        continue
                    # groundtruth:[person..[keypoints..[x, y, Z, score(0:None, 1:invisible, 2:visible), X, Y, Z,
                    #                                   f_x, f_y, cx, cy]]]
                    if len(gt_bodys[0][0]) < 11:
                        scale['f_x'] = gt_bodys[0, 0, 7]
                        scale['f_y'] = gt_bodys[0, 0, 7]
                        scale['cx'] = scale['img_width']/2
                        scale['cy'] = scale['img_height']/2
                    else:
                        scale['f_x'] = gt_bodys[0, 0, 7]
                        scale['f_y'] = gt_bodys[0, 0, 8]
                        scale['cx'] = gt_bodys[0, 0, 9]
                        scale['cy'] = gt_bodys[0, 0, 10]
                else:
                    gt_bodys = None
                    # use default values
                    scale = {k: scales[k][i].numpy() for k in scales}
                    scale['f_x'] = scale['img_width']
                    scale['f_y'] = scale['img_width']
                    scale['cx'] = scale['img_width']/2
                    scale['cy'] = scale['img_height']/2

                hmsIn = outputs_2d[i]

                # if the first pair is [1, 0], uncomment the code below
                # hmsIn[cfg.DATASET.KEYPOINT.NUM:cfg.DATASET.KEYPOINT.NUM+2] *= -1
                # outputs_3d[i, 0] *= -1

                hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255
                hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127
                rDepth = outputs_rd[i][0]
                # no batch implementation yet
                pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True)
                if len(pred_bodys_2d) > 0:
                    pred_bodys_2d[:, :, :2] *= cfg.dataset.STRIDE  # resize poses to the input-net shape
                    pred_bodys_2d = pred_bodys_2d.numpy()

                pafs_3d = outputs_3d[i].numpy().transpose(1, 2, 0)
                root_d = outputs_rd[i][0].numpy()

                paf_3d_upsamp = cv2.resize(
                    pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)
                root_d_upsamp = cv2.resize(
                    root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)

                # generate 3d prediction bodys
                pred_bodys_2d = register_pred(pred_bodys_2d, gt_bodys)

                if len(pred_bodys_2d) == 0:
                    continue
                pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scale)
                pred_bodys_3d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scale)

                if refine_model is not None:
                    new_pred_bodys_3d = lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, 
                                                                device=device, root_n=cfg.DATASET.ROOT_IDX)
                else:
                    new_pred_bodys_3d = pred_bodys_3d

                if cfg.TEST_MODE == "generate_train":
                    save_result_for_train_refine(pred_bodys_2d, new_pred_bodys_3d, gt_bodys, pred_rdepths, result)
                else:
                    save_result(pred_bodys_2d, new_pred_bodys_3d, gt_bodys, pred_rdepths, img_path[i], result)

    dir_name = os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]
    pair_file_name = os.path.join(output_dir, '{}_{}_{}_{}.json'.format(dir_name, cfg.TEST_MODE,
                                                                        cfg.DATA_MODE, cfg.JSON_SUFFIX_NAME))
    with open(pair_file_name, 'w') as f:
        json.dump(result, f)
    logger.info("Pairs writed to {}".format(pair_file_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", "-t", type=str, default="run_inference",
                        choices=['generate_train', 'generate_result', 'run_inference'],
                        help='Type of test. One of "generate_train": generate refineNet datasets, '
                             '"generate_result": save inference result and groundtruth, '
                             '"run_inference": save inference result for input images.')
    parser.add_argument("--data_mode", "-d", type=str, default="test",
                        choices=['test', 'generation'],
                        help='Only used for "generate_train" test_mode, "generation" for refineNet train dataset,'
                             '"test" for refineNet test dataset.')
    parser.add_argument("--SMAP_path", "-p", type=str, default='log/SMAP.pth',
                        help='Path to SMAP model')
    parser.add_argument("--RefineNet_path", "-rp", type=str, default='',
                        help='Path to RefineNet model, empty means without RefineNet')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='Batch_size of test')
    parser.add_argument("--do_flip", type=float, default=0,
                        help='Set to 1 if do flip when test')
    parser.add_argument("--dataset_path", type=str, default="",
                        help='Image dir path of "run_inference" test mode')
    parser.add_argument("--json_name", type=str, default="",
                        help='Add a suffix to the result json.')
    args = parser.parse_args()
    cfg.TEST_MODE = args.test_mode
    cfg.DATA_MODE = args.data_mode
    cfg.REFINE = len(args.RefineNet_path) > 0
    cfg.DO_FLIP = args.do_flip
    cfg.JSON_SUFFIX_NAME = args.json_name
    cfg.TEST.IMG_PER_GPU = args.batch_size

    os.makedirs(cfg.TEST_DIR, exist_ok=True)
    logger = get_logger(
            cfg.DATASET.NAME, cfg.TEST_DIR, 0, 'test_log_{}.txt'.format(args.test_mode))

    model = SMAP(cfg, run_efficient=cfg.RUN_EFFICIENT)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if args.test_mode == "run_inference":
        test_dataset = CustomDataset(cfg, args.dataset_path)
        data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        data_loader = get_test_loader(cfg, num_gpu=1, local_rank=0, stage=args.data_mode)

    if cfg.REFINE:
        refine_model = RefineNet()
        refine_model.to(device)
        refine_model_file = args.RefineNet_path
    else:
        refine_model = None
        refine_model_file = ""

    model_file = args.SMAP_path
    if os.path.exists(model_file):
        state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = state_dict['model']
        model.load_state_dict(state_dict)
        if os.path.exists(refine_model_file):
            refine_model.load_state_dict(torch.load(refine_model_file))
        elif refine_model is not None:
            logger.info("No such RefineNet checkpoint of {}".format(args.RefineNet_path))
            return
        generate_3d_point_pairs(model, refine_model, data_loader, cfg, logger, device,
                                output_dir=os.path.join(cfg.OUTPUT_DIR, "result"))
    else:
        logger.info("No such checkpoint of SMAP {}".format(args.SMAP_path))


if __name__ == '__main__':
    main()
