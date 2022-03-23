"""
@author: Jianan Zhen
@contact: zhenjianan@sensetime.com
"""
import numpy as np
import copy
import torch

from config import cfg
from lib.utils.post_3d import decode_pose, chain_bones, get_3d_points


joint_to_limb_heatmap_relationship = [[1, 0], [0, 2],
                                      [0, 9], [9, 10], [10, 11],
                                      [0, 3], [3, 4], [4, 5],
                                      [2, 12], [12, 13], [13, 14],
                                      [2, 6], [6, 7], [7, 8]]
paf_z_coords_per_limb = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
bone_length = np.asarray([26.42178982, 48.36980909,
                          14.88291009, 31.28002332, 23.915707,
                          14.97674918, 31.28002549, 23.91570732,
                          12.4644364,  48.26604433, 39.03553194,
                          12.4644364, 48.19076948, 39.03553252])
NUM_LIMBS = len(joint_to_limb_heatmap_relationship)
test_root_n = 2


def distance(a_x, a_y, b_x, b_y):
    return ((a_x - b_x)**2 + (a_y - b_y)**2)**0.5


def chain_2d(candidate, subset):
    if len(subset) == 0:
        return np.asarray([])
    # register every person
    pred_bodys = np.zeros((len(subset), cfg.DATASET.KEYPOINT.NUM, 4), np.float)
    for body_i in range(len(subset)):
        for j in range(cfg.DATASET.KEYPOINT.NUM):
            index = int(subset[body_i][j])
            if index >= 0:
                pred_bodys[body_i][j][0] = candidate[index][0]
                pred_bodys[body_i][j][1] = candidate[index][1]
                pred_bodys[body_i][j][3] = candidate[index][2]
    return pred_bodys


def register_pred(pred_bodys, gt_bodys, root_n=2):
    if len(pred_bodys) == 0:
        return np.zeros((len(gt_bodys), cfg.DATASET.KEYPOINT.NUM, 4), np.float), 1
    if gt_bodys is not None:
        distance_array = np.ones((len(gt_bodys), len(pred_bodys)), dtype=np.float) * 2 ** 15
        norm_head_size = np.zeros(len(gt_bodys), dtype=np.float)
        for i in range(len(gt_bodys)):
            gt_neck = gt_bodys[i][0]
            gt_head = gt_bodys[i][1]
            for j in range(len(pred_bodys)):
                d = distance(gt_bodys[i][root_n][0], gt_bodys[i][root_n][1],
                             pred_bodys[j][root_n][0], pred_bodys[j][root_n][1])
                distance_array[i][j] = d
            norm_head_size[i] = distance(gt_head[0], gt_head[1], gt_neck[0], gt_neck[1]) / 3
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
        new_pred_bodys = np.zeros((len(gt_bodys), cfg.DATASET.KEYPOINT.NUM, 4), np.float)
        for i in range(len(gt_bodys)):
            if corres[i] >= 0:
                new_pred_bodys[i] = pred_bodys[corres[i]]
    else:
        new_pred_bodys = list()
        for pred_body in pred_bodys:
            if pred_body[root_n][3] != 0:
                new_pred_bodys.append(pred_body)
        new_pred_bodys = np.asarray(new_pred_bodys)
        norm_head_size = []
        if len(new_pred_bodys) == 0:
            new_pred_bodys = []
    return new_pred_bodys, norm_head_size


def eval_one_image(pred_bodys, gt_bodys, output, root_n=2):
    # remove person whose root was blocked
    keypoint_num = len(gt_bodys[0])
    count_pred = np.zeros(keypoint_num, dtype=np.int)
    count_gt = np.zeros(keypoint_num, dtype=np.int)
    distance_e = np.zeros(keypoint_num, dtype=np.float)
    pred_bodys, norm_head_size = register_pred(pred_bodys, gt_bodys)
    if len(pred_bodys) != len(gt_bodys):
        return pred_bodys, norm_head_size
    if output is not None:
        # compute error
        for i, gt_body in enumerate(gt_bodys):
            if pred_bodys[i][root_n][0] > 0 and pred_bodys[i][root_n][1] > 0:
                pred_body = pred_bodys[i]
                for j in range(keypoint_num):
                    if gt_body[j][3] > 1:
                        dis = distance(gt_body[j][0], gt_body[j][1], pred_body[j][0], pred_body[j][1])
                        if dis < norm_head_size[i]:
                            distance_e[j] += dis / norm_head_size[i]
                            count_pred[j] += 1
                        count_gt[j] += 1

        output['count_gt'] += count_gt
        output['count_pred'] += count_pred
        output['distance_e'] += distance_e
    return pred_bodys, norm_head_size


def generate_rootZ(pred_bodys, gt_bodys, error, paf_3d_upsamp, root_d_upsamp, scale,
                   test_mode, num_intermed_pts=10, root_n=2):
    limb_intermed_coords = np.empty((2, num_intermed_pts), dtype=np.intp)
    depth_v = np.zeros((len(pred_bodys), NUM_LIMBS), dtype=np.float)
    depth_necks_pred = np.zeros(len(pred_bodys), dtype=np.float)
    distance_d = np.zeros(NUM_LIMBS, dtype=np.float)
    reverse_count = np.zeros(NUM_LIMBS, dtype=np.float)
    count_pred_bone = np.zeros(NUM_LIMBS, dtype=np.int)
    for i, pred_body in enumerate(pred_bodys):
        if pred_body[root_n][3] > 0:
            depth_necks_pred[i] = root_d_upsamp[int(pred_body[root_n][1]), int(pred_body[root_n][0])] * scale['scale'] * scale['f_x']
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
                    if test_mode == 'eval':
                        real_depth = gt_bodys[i][bone[1]][2] - gt_bodys[i][bone[0]][2]
                        distance_d[k] += abs(mean_val - real_depth)
                        count_pred_bone[k] += 1
                        if mean_val * real_depth < -1:
                            reverse_count[k] += 1
            chain_bones(pred_bodys, depth_v, i, depth_0=0)

    if test_mode == 'eval':
        error['distance_d'] += distance_d
        error['reverse_count'] += reverse_count
        error['count_pred_bone'] += count_pred_bone
    return depth_necks_pred


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


def lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, root_n=2):
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
    inp = torch.from_numpy(input_point).float().cuda()
    pred = refine_model(inp)
    pred = pred.cpu().numpy()
    pred = np.resize(pred, (pred.shape[0], 15, 3))
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if j != root_n: #and pred_bodys_3d[i, j, 3] == 0:
                pred[i, j] += pred_bodys_3d[i, root_n, :3]
            else:
                pred[i, j] = pred_bodys_3d[i, j, :3]
    pred = np.concatenate([pred, score_after_refine], axis=2)
    return pred


def refine_3d_pose(pred_bodys, refine_model, root_n=2):
    root_3d_bodys = copy.deepcopy(pred_bodys)
    score_after_refine = np.ones([pred_bodys.shape[0], pred_bodys.shape[1], 1], dtype=np.float)
    for i in range(len(root_3d_bodys)):
        if root_3d_bodys[i, root_n, 3] == 0:
            score_after_refine[i] = 0
        for j in range(len(root_3d_bodys[0])):
            if j == root_n:
                continue
            if root_3d_bodys[i, j, 3] > 0:
                root_3d_bodys[i, j, :3] -= root_3d_bodys[i, root_n, :3]

    """
    for i in range(len(root_2d_bodys)):
        root_2d_bodys[i, :, :2] *= depth_necks[i]
    """
    root_3d_bodys = np.resize(root_3d_bodys[:, :, :3], (root_3d_bodys.shape[0], 45))
    #root_3d_bodys = np.resize(root_3d_bodys, (root_3d_bodys.shape[0], 60))
    inp = torch.from_numpy(root_3d_bodys).float().cuda()
    pred = refine_model(inp)
    pred = pred.cpu().numpy()
    pred = np.resize(pred, (pred.shape[0], 15, 3))
    # At present, RefineNet is only used for repair
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if j != root_n and pred_bodys[i, j, 3] == 0:
                pred[i, j] += pred_bodys[i, root_n, :3]
            else:
                pred[i, j] = pred_bodys[i, j, :3]

    pred = np.concatenate([pred, score_after_refine], axis=2)

    return pred


def save_result_for_train_refine(pred_bodys_2d, pred_bodys_3d, gt_bodys, depth_necks_pred,
                                 result, root_n=2):
    for i, pred_body in enumerate(pred_bodys_3d):
        if pred_body[root_n][3] != 0:
            pair = {}
            pair['pred_3d'] = pred_body.tolist()
            pair['pred_2d'] = pred_bodys_2d[i].tolist()
            pair['gt_3d'] = gt_bodys[i][:, 4:7].tolist()
            pair['root_d'] = depth_necks_pred[i]
            result['3d_pairs'].append(pair)


def save_result_for_eval(pred_bodys_2d, pred_bodys_3d, img_path, result, root_n=2):
    bodys = list()
    for index, pred_body_2d in enumerate(pred_bodys_2d):
        if pred_body_2d[root_n, 3] == 0:
            continue
        body = dict()
        body['joints_3d_cam'] = pred_bodys_3d[index]
        body['joints_3d_cam'] = np.ravel(body['joints_3d_cam'])
        body['joints_3d_cam'] = body['joints_3d_cam'].tolist()

        body['score'] = np.max(pred_bodys_2d[index, :, 3]) + np.mean(pred_bodys_2d[index, :, 3])
        pred_x = pred_bodys_2d[index, :, 0]
        pred_y = pred_bodys_2d[index, :, 1]
        min_x = np.min(pred_x[pred_x > 0])
        max_x = np.max(pred_x[pred_x > 0])
        min_y = np.min(pred_y[pred_y > 0])
        max_y = np.max(pred_y[pred_y > 0])
        body['bbox'] = [min_x*2, min_y*2, (max_x-min_x)*2, (max_y-min_y)*2]
        bodys.append(body)

    image_name = img_path.tostring().decode(encoding='utf-8').strip().split('/', 1)[-1]
    result[image_name] = bodys


def eval_3d(pred_bodys_3d, gt_bodys, error, key_word='', root_n=2):
    for i, pred_body in enumerate(pred_bodys_3d):
        if gt_bodys[i][root_n][3] < 2:
            continue
        error['total_people_gt'] += 1
        if pred_bodys_3d[i][root_n][3] == 0:
            continue
        gt_body = gt_bodys[i, :, 4:7]
        root_pred_body = copy.deepcopy(pred_body[:, :3])
        root_pred_body[:, :] -= root_pred_body[test_root_n, :]
        root_gt_body = copy.deepcopy(gt_body)
        root_gt_body[:, :] -= root_gt_body[test_root_n, :]
        error_i = np.linalg.norm(np.abs(pred_body[:, :3] - gt_body), axis=1)
        error_i[pred_body[:, 3] == 0] = 0
        real_PCK = np.asarray(error_i < 15).astype(np.int)
        real_PCK[pred_body[:, 3] == 0] = 0
        if error_i[0] < 15:
            error['less_15'+key_word] += 1
        root_error_i = np.linalg.norm(np.abs(root_gt_body - root_pred_body), axis=1)
        root_error_i[pred_body[:, 3] == 0] = 0
        root_PCK = np.asarray(root_error_i < 15).astype(np.int)
        root_PCK[pred_body[:, 3] == 0] = 0
        count_point = np.ones(15, dtype=np.float)
        count_point[pred_body[:, 3] == 0] = 0
        if i + 1 < len(pred_bodys_3d) and i + 1 < len(gt_bodys) and pred_bodys_3d[i + 1][root_n][0] != 0:
            error['total_pair_count'+key_word] += 1
            if (gt_body[root_n][2] - gt_bodys[i + 1][root_n][6]) * (pred_body[root_n][2] - pred_bodys_3d[i + 1][root_n][2]) < 0:
                error['reverse_pair_count'+key_word] += 1

        error['count_point'+key_word] += count_point
        error['real_error'+key_word] += error_i
        error['real_PCK'+key_word] += real_PCK
        error['root_error'+key_word] += root_error_i
        error['root_PCK'+key_word] += root_PCK
        error['count_people'+key_word] += 1


def save_result(pred_bodys_2d, pred_bodys_3d, gt_bodys, depth_necks_pred, img_path, result, error,
                refine_model):
    if gt_bodys is not None:
        if refine_model is not None:
            eval_3d(pred_bodys_3d, gt_bodys, error, key_word='_after_refine')
        else:
            eval_3d(pred_bodys_3d, gt_bodys, error, key_word='')

    pair = dict()
    pair['pred_2d'] = pred_bodys_2d.tolist()
    pair['pred_3d'] = pred_bodys_3d.tolist()
    pair['root_d'] = depth_necks_pred.tolist()
    pair['image_path'] = img_path.tostring().decode(encoding='utf-8').strip()
    if gt_bodys is not None:
        pair['gt_3d'] = gt_bodys[:, :, 4:].tolist()
        pair['gt_2d'] = gt_bodys[:, :, :4].tolist()
    else:
        pair['gt_3d'] = list()
        pair['gt_2d'] = list()
    result['3d_pairs'].append(pair)


def initialization(error, cfg):
    if cfg.REFINE:
        key = '_after_refine'
    else:
        key = ''
    if cfg.TEST_MODE == 'eval':
        error['count_gt'] = np.zeros(cfg.DATASET.KEYPOINT.NUM, dtype=np.int)
        error['count_pred'] = np.zeros(cfg.DATASET.KEYPOINT.NUM, dtype=np.int)
        error['distance_e'] = np.zeros(cfg.DATASET.KEYPOINT.NUM, dtype=np.float)
        error['distance_d'] = np.zeros(NUM_LIMBS, dtype=np.float)
        error['reverse_count'] = np.zeros(NUM_LIMBS, dtype=np.float)
        error['count_pred_bone'] = np.zeros(NUM_LIMBS, dtype=np.int)

    if cfg.TEST_MODE == 'generate_result' or cfg.TEST_MODE == 'eval':
        error['real_error'+key] = np.zeros(15, dtype=np.float)
        error['root_error'+key] = np.zeros(15, dtype=np.float)
        error['count_people'+key] = 0
        error['total_people_gt'] = 0
        error['count_point'+key] = np.zeros(15, dtype=np.float)
        error['total_pair_count'+key] = 1e-8
        error['reverse_pair_count'+key] = 0.0
        error['less_15'+key] = 0.0
        error['root_PCK'+key] = np.zeros(15, dtype=np.float)
        error['real_PCK'+key] = np.zeros(15, dtype=np.float)


def calculate_and_log(test_mode, error, result, logger, iteration):
    if test_mode == 'eval':
        error_point = np.zeros(cfg.DATASET.KEYPOINT.NUM, dtype=np.float)
        depth_e = np.zeros(NUM_LIMBS, dtype=np.float)
        depth_reverse_count = np.zeros(NUM_LIMBS, dtype=np.float)
        recall = np.zeros(cfg.DATASET.KEYPOINT.NUM, dtype=np.float)
        mask = error['count_pred'] > 0
        mask_bone = error['count_pred_bone'] > 0
        error_point[mask] = error['distance_e'][mask] / error['count_pred'][mask]
        depth_e[mask_bone] = error['distance_d'][mask_bone] / error['count_pred_bone'][mask_bone]
        depth_reverse_count[mask_bone] = error['reverse_count'][mask_bone] / error['count_pred_bone'][mask_bone]
        avg_error = np.average(error_point[mask])
        mask = error['count_gt'] > 0
        recall[mask] = error['count_pred'][mask] / error['count_gt'][mask]
        avg_recall = np.average(recall[mask])
        logger.info("In iteration {}".format(iteration))
        logger.info("keypoints error of validation dataset is {}".format(error_point))
        logger.info("Keypoints recall of validation dataset is {}".format(recall))
        logger.info("Bones depth distance is {}".format(depth_e))
        logger.info("Bones reverse count is {}".format(depth_reverse_count))
        logger.info("Average error is {}, average recall is {}".format(avg_error, avg_recall))
    if test_mode == 'generate_result' or test_mode == 'eval':
        if cfg.REFINE:
            key = '_after_refine'
        else:
            key = ''
        error['real_error' + key] = (error['real_error' + key] / error['count_point' + key])
        error['root_error' + key] = (error['root_error' + key] / error['count_point' + key])
        error['root_PCK' + key] = (error['root_PCK' + key] / error['count_people' + key])
        error['real_PCK' + key] = (error['real_PCK' + key] / error['count_people' + key])
        logger.info("Real error is {}, root error is {}".format(error['real_error' + key],
                                                                error['root_error' + key]))
        logger.info("Real PCK_25 is {}, root PCK_15 is {}. ".format(error['real_PCK' + key],
                                                                    error['root_PCK' + key]))
        logger.info("Recall of points is {}".format(error['count_point' + key] /
                                                    error['count_people' + key]))
        logger.info("Reverse rate is {}".format(error['reverse_pair_count' + key] /
                                                error['total_pair_count' + key]))
        logger.info("Less than 25 rate is {}".format(error['less_15' + key] /
                                                     error['total_people_gt']))
        logger.info("find {} people, total is {}, recall is {}".format(error['count_people' + key],
                                                                       error['total_people_gt'],
                                                                       error['count_people' + key]/error['total_people_gt']))

        error['real_error' + key] = error['real_error' + key].tolist()
        error['root_error' + key] = error['root_error' + key].tolist()
        error['count_point' + key] = error['count_point' + key].tolist()
        error['real_PCK' + key] = error['real_PCK' + key].tolist()
        error['root_PCK' + key] = error['root_PCK' + key].tolist()
        if test_mode == 'generate_result':
            result['error'] = error
