import cv2
import numpy as np


def generate_heatmap(bodys, output_shape, stride, keypoint_num, kernel=(7, 7)):
    heatmaps = np.zeros((keypoint_num, *output_shape), dtype='float32')

    for i in range(keypoint_num):
        for j in range(len(bodys)):
            if bodys[j][i][3] < 1:
                continue
            target_y = bodys[j][i][1] / stride
            target_x = bodys[j][i][0] / stride
            heatmaps[i, int(target_y), int(target_x)] = 1
        heatmaps[i] = cv2.GaussianBlur(heatmaps[i], kernel, 0)
        maxi = np.amax(heatmaps[i])
        if maxi <= 1e-8:
            continue
        heatmaps[i] /= maxi / 255

    return heatmaps

def generate_rdepth(meta, stride, root_idx, max_people):
    bodys = meta['bodys']
    scale = meta['scale']
    rdepth = np.zeros((max_people, 3), dtype='float32')
    for j in range(len(bodys)):
        if bodys[j][root_idx, 3] < 1 or j >= max_people:
            continue
        rdepth[j, 0] = bodys[j][root_idx, 1] / stride
        rdepth[j, 1] = bodys[j][root_idx, 0] / stride
        rdepth[j, 2] = bodys[j][root_idx, 2] / bodys[j][root_idx, 7] / scale  # normalize by f and scale
    rdepth = rdepth[np.argsort(-rdepth[:, 2])]
    return rdepth

def generate_paf(bodys, output_shape, params_transform, paf_num, paf_vector, paf_thre, with_mds):
    pafs = np.zeros((paf_num * 3, *output_shape), dtype='float32')
    count = np.zeros((paf_num, *output_shape), dtype='float32')
    for i in range(paf_num):
        for j in range(len(bodys)):
            if paf_thre > 1 and with_mds:
                if bodys[j][paf_vector[i][0]][3] < 2 or bodys[j][paf_vector[i][1]][3] < 2:
                    continue
            elif bodys[j][paf_vector[i][0]][3] < 1 or bodys[j][paf_vector[i][1]][3] < 1:
                continue
            centerA = np.array(bodys[j][paf_vector[i][0]][:3], dtype=int)
            centerB = np.array(bodys[j][paf_vector[i][1]][:3], dtype=int)
            pafs[i*3:i*3+3], count[i] = putVecMaps3D(centerA, centerB, pafs[i*3:i*3+3], count[i], \
                                                     params_transform, paf_thre)
    pafs[0::3] *= 127
    pafs[1::3] *= 127

    return pafs

def putVecMaps3D(centerA, centerB, accumulate_vec_map, count, params_transform, thre):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)
    z_A = centerA[2]
    z_B = centerB[2]
    centerA = centerA[:2]
    centerB = centerB[:2]

    stride = params_transform['stride']
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    limb_z = z_B - z_A
    norm = np.linalg.norm(limb_vec)
    if norm < 1.0:  # limb is too short, ignore it
        return accumulate_vec_map, count

    limb_vec_unit = limb_vec / norm

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[:, yy, xx] = np.repeat(mask[np.newaxis, :, :], 3, axis=0)
    vec_map[:2, yy, xx] *= limb_vec_unit[:, np.newaxis, np.newaxis]
    vec_map[2, yy, xx] *= limb_z
    mask = np.logical_or.reduce(
        (np.abs(vec_map[0, :, :]) != 0, np.abs(vec_map[1, :, :]) != 0))

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[np.newaxis, :, :])
    accumulate_vec_map += vec_map

    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[np.newaxis, :, :])
    count[mask == True] = 0

    return accumulate_vec_map, count

def putVecMaps(centerA, centerB, accumulate_vec_map, count, params_transform, thre):
    """Implement Part Affinity Fields
    :param centerA: int with shape (2,) or (3,), centerA will pointed by centerB.
    :param centerB: int with shape (2,) or (3,), centerB will point to centerA.
    :param accumulate_vec_map: one channel of paf.
    :param count: store how many pafs overlaped in one coordinate of accumulate_vec_map.
    :param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
    """

    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    stride = params_transform['stride']
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if norm < 1.0:
        # print 'limb is too short, ignore it...'
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm
    # print 'limb unit vector: {}'.format(limb_vec_unit)

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[:, yy, xx] = np.repeat(mask[np.newaxis, :, :], 2, axis=0)
    vec_map[:, yy, xx] *= limb_vec_unit[:, np.newaxis, np.newaxis]

    mask = np.logical_or.reduce(
        (np.abs(vec_map[0, :, :]) > 0, np.abs(vec_map[1, :, :]) > 0))

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[np.newaxis, :, :])
    accumulate_vec_map += vec_map
    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[np.newaxis, :, :])
    count[mask == True] = 0

    return accumulate_vec_map, count


