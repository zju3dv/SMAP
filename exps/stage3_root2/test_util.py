"""
@author: Jianan Zhen
@contact: zhenjianan@sensetime.com
"""
import numpy as np
import copy
import torch

from config import cfg
from lib.utils.post_3d import get_3d_points


joint_to_limb_heatmap_relationship = cfg.DATASET.PAF.VECTOR
paf_z_coords_per_limb = list(range(cfg.DATASET.KEYPOINT.NUM))
NUM_LIMBS = len(joint_to_limb_heatmap_relationship)


def register_pred(pred_bodys, gt_bodys, root_n=2):
    if len(pred_bodys) == 0:
        return np.asarray([])
    if gt_bodys is not None:
        root_gt = gt_bodys[:, root_n, :2]
        root_pd = pred_bodys[:, root_n, :2]
        distance_array = np.linalg.norm(root_gt[:, None, :] - root_pd[None, :, :], axis=2)
        corres = np.ones(len(gt_bodys), np.int) * -1
        occupied = np.zeros(len(pred_bodys), np.int)
        while np.min(distance_array) < 30:
            min_idx = np.where(distance_array == np.min(distance_array))
            for i in range(len(min_idx[0])):
                distance_array[min_idx[0][i]][min_idx[1][i]] = 50
                if corres[min_idx[0][i]] >= 0 or occupied[min_idx[1][i]]:
                    continue
                else:
                    corres[min_idx[0][i]] = min_idx[1][i]
                    occupied[min_idx[1][i]] = 1
        new_pred_bodys = np.zeros((len(gt_bodys), len(gt_bodys[0]), 4), np.float)
        for i in range(len(gt_bodys)):
            if corres[i] >= 0:
                new_pred_bodys[i] = pred_bodys[corres[i]]
    else:
        new_pred_bodys = pred_bodys[pred_bodys[:, root_n, 3] != 0]
    return new_pred_bodys


def chain_bones(pred_bodys, depth_v, i, depth_0=0, root_n=2):
    if root_n == 2:
        start_number = 2
        pred_bodys[i][2][2] = depth_0
        pred_bodys[i][0][2] = pred_bodys[i][2][2] - depth_v[i][1]
    else:
        start_number = 1
        pred_bodys[i][0][2] = depth_0
    pred_bodys[i][1][2] = pred_bodys[i][0][2] + depth_v[i][0]
    for k in range(start_number, NUM_LIMBS):
        src_k = joint_to_limb_heatmap_relationship[k][0]
        dst_k = joint_to_limb_heatmap_relationship[k][1]
        pred_bodys[i][dst_k][2] = pred_bodys[i][src_k][2] + depth_v[i][k]


def generate_relZ(pred_bodys, paf_3d_upsamp, root_d_upsamp, scale, num_intermed_pts=10, root_n=2):
    limb_intermed_coords = np.empty((2, num_intermed_pts), dtype=np.intp)
    depth_v = np.zeros((len(pred_bodys), NUM_LIMBS), dtype=np.float)
    depth_roots_pred = np.zeros(len(pred_bodys), dtype=np.float)
    for i, pred_body in enumerate(pred_bodys):
        if pred_body[root_n][3] > 0:
            depth_roots_pred[i] = root_d_upsamp[int(pred_body[root_n][1]), int(pred_body[root_n][0])] * scale['scale'] * scale['f_x']
            for k, bone in enumerate(joint_to_limb_heatmap_relationship):
                joint_src = pred_body[bone[0]]
                joint_dst = pred_body[bone[1]]
                if joint_dst[3] > 0 and joint_src[3] > 0:
                    depth_idx = paf_z_coords_per_limb[k]
                    # Linearly distribute num_intermed_pts points from the x
                    # coordinate of joint_src to the x coordinate of joint_dst
                    limb_intermed_coords[1, :] = np.round(np.linspace(
                        joint_src[0], joint_dst[0], num=num_intermed_pts))
                    limb_intermed_coords[0, :] = np.round(np.linspace(
                        joint_src[1], joint_dst[1], num=num_intermed_pts))  # Same for the y coordinate
                    intermed_paf = paf_3d_upsamp[limb_intermed_coords[0, :],
                                                 limb_intermed_coords[1, :], depth_idx]
                    min_val, max_val = np.percentile(intermed_paf, [10, 90])
                    intermed_paf[intermed_paf < min_val] = min_val
                    intermed_paf[intermed_paf > max_val] = max_val
                    mean_val = np.mean(intermed_paf)
                    depth_v[i][k] = mean_val
            chain_bones(pred_bodys, depth_v, i, depth_0=0)
    return depth_roots_pred


def gen_3d_pose(pred_bodys, depth_necks, scale):
    bodys = copy.deepcopy(pred_bodys)
    bodys[:, :, 0] = bodys[:, :, 0]/scale['scale'] - (scale['net_width']/scale['scale']-scale['img_width'])/2
    bodys[:, :, 1] = bodys[:, :, 1]/scale['scale'] - (scale['net_height']/scale['scale']-scale['img_height'])/2
    K = np.asarray([[scale['f_x'], 0, scale['cx']], [0, scale['f_y'], scale['cy']], [0, 0, 1]])
    bodys_3d = get_3d_points(bodys, depth_necks, K)
    for i in range(bodys_3d.shape[0]):
        for j in range(bodys_3d.shape[1]):
            if bodys_3d[i, j, 3] == 0:
                bodys_3d[i, j] = 0
    return bodys_3d


def lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, device, root_n=2):
    root_3d_bodys = copy.deepcopy(pred_bodys_3d)
    root_2d_bodys = copy.deepcopy(pred_bodys_2d)
    score_after_refine = np.ones([pred_bodys_3d.shape[0], pred_bodys_3d.shape[1], 1], dtype=np.float)
    input_point = np.zeros((pred_bodys_3d.shape[0], 15, 5), dtype=np.float)
    input_point[:, root_n, :2] = root_2d_bodys[:, root_n, :2]
    input_point[:, root_n, 2:] = root_3d_bodys[:, root_n, :3]
    for i in range(len(root_3d_bodys)):
        if root_3d_bodys[i, root_n, 3] == 0:
            score_after_refine[i] = 0
        for j in range(len(root_3d_bodys[0])):
            if j != root_n and root_3d_bodys[i, j, 3] > 0:
                input_point[i, j, :2] = root_2d_bodys[i, j, :2] - root_2d_bodys[i, root_n, :2]
                input_point[i, j, 2:] = root_3d_bodys[i, j, :3] - root_3d_bodys[i, root_n, :3]
    input_point = np.resize(input_point, (input_point.shape[0], 75))
    inp = torch.from_numpy(input_point).float().to(device)
    pred = refine_model(inp)
    if pred.device.type == 'cuda':
        pred = pred.cpu().numpy()
    else:
        pred = pred.numpy()
    pred = np.resize(pred, (pred.shape[0], 15, 3))
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if j != root_n: #and pred_bodys_3d[i, j, 3] == 0:
                pred[i, j] += pred_bodys_3d[i, root_n, :3]
            else:
                pred[i, j] = pred_bodys_3d[i, j, :3]
    pred = np.concatenate([pred, score_after_refine], axis=2)
    return pred


def save_result_for_train_refine(pred_bodys_2d, pred_bodys_3d, gt_bodys, pred_rdepths,
                                 result, root_n=2):
    for i, pred_body in enumerate(pred_bodys_3d):
        if pred_body[root_n][3] != 0:
            pair = {}
            pair['pred_3d'] = pred_body.tolist()
            pair['pred_2d'] = pred_bodys_2d[i].tolist()
            pair['gt_3d'] = gt_bodys[i][:, 4:7].tolist()
            pair['root_d'] = pred_rdepths[i]
            result['3d_pairs'].append(pair)


def save_result(pred_bodys_2d, pred_bodys_3d, gt_bodys, pred_rdepths, img_path, result):
    pair = dict()
    pair['pred_2d'] = pred_bodys_2d.tolist()
    pair['pred_3d'] = pred_bodys_3d.tolist()
    pair['root_d'] = pred_rdepths.tolist()
    pair['image_path'] = img_path
    if gt_bodys is not None:
        pair['gt_3d'] = gt_bodys[:, :, 4:].tolist()
        pair['gt_2d'] = gt_bodys[:, :, :4].tolist()
    else:
        pair['gt_3d'] = list()
        pair['gt_2d'] = list()
    result['3d_pairs'].append(pair)
