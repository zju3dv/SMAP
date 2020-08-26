# coding=utf-8

import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage


"""The purpose of Augmentor is to automate image augmentation 
   in order to expand datasets as input for our algorithms.
:aut_scale : Scales them by dice2 (<1, so it is zoom out). 
:aug_croppad centerB: int with shape (2,), centerB will point to centerA.
:aug_flip: Mirrors the image around a vertical line running through its center.
:aug_rotate: Rotates the image. The angle of rotation, in degrees, 
             is specified by a random integer value that is included
             in the transform argument.
             
:param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
"""


def aug_scale(meta, img, params_transform, mask=None):
    dice = random.random()  # [0,1)
    if dice > params_transform['scale_prob']:
        scale_multiplier = 1
    else:
        dice2 = random.random()
        # linear shear into [scale_min, scale_max]
        scale_multiplier = (
            params_transform['scale_max'] - params_transform['scale_min']) * dice2 + \
            params_transform['scale_min']
    scale_abs = params_transform['target_dist'] / meta['scale_provided']
    scale = scale_abs * scale_multiplier
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)
    if mask is not None:
        mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # modify meta data
    meta['objpos'] *= scale
    meta['joint_self'][:, :2] *= scale
    if meta['numOtherPeople'] != 0:
        meta['objpos_other'] *= scale
        meta['joint_others'][:, :, :2] *= scale
    if mask is not None:
        return meta, img, mask
    else:
        return meta, img


def aug_croppad(meta, img, params_transform, with_augmentation=True):
    dice_x = random.random()
    dice_y = random.random()
    scale_random = random.random()
    scale_multiplier = ((params_transform['scale_max'] - params_transform['scale_min']) *
                        scale_random + params_transform['scale_min'])
    crop_x = int(params_transform['crop_size_x'])
    crop_y = int(params_transform['crop_size_y'])
   
    scale = min(params_transform['crop_size_x'] / float(img.shape[1]),
                params_transform['crop_size_y'] / float(img.shape[0]))
                
    if with_augmentation:
        scale *= scale_multiplier
    
    meta['scale'] = scale
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    meta['bodys'][:, :, :2] *= scale
    # meta['center'] *= scale

    x_offset = int((dice_x - 0.5) * 2 *
                   params_transform['center_perterb_max'])
    y_offset = int((dice_y - 0.5) * 2 *
                   params_transform['center_perterb_max'])

    center = meta['center'] * scale + np.array([x_offset, y_offset])
    center = center.astype(int)

    # pad up and down
    pad_v = np.ones((crop_y, img.shape[1], 3), dtype=np.uint8) * 128
    img = np.concatenate((pad_v, img, pad_v), axis=0)

    # pad right and left
    pad_h = np.ones((img.shape[0], crop_x, 3), dtype=np.uint8) * 128
    img = np.concatenate((pad_h, img, pad_h), axis=1)

    img = img[int(center[1] + crop_y / 2):int(center[1] + crop_y / 2 + crop_y),
              int(center[0] + crop_x / 2):int(center[0] + crop_x / 2 + crop_x), :]

    offset_left = crop_x / 2 - center[0]
    offset_up = crop_y / 2 - center[1]

    offset = np.array([offset_left, offset_up], np.int)
    meta['center'] += offset
    for i in range(len(meta['bodys'])):
        meta['bodys'][i][:, :2] += offset
        mask = np.logical_or.reduce((meta['bodys'][i][:, 0] >= crop_x,
                                     meta['bodys'][i][:, 0] < 0,
                                     meta['bodys'][i][:, 1] >= crop_y,
                                     meta['bodys'][i][:, 1] < 0))

        meta['bodys'][i][mask == True, 3] = 0

    return meta, img


def aug_flip(meta, img, params_transform):
    dice = random.random()
    doflip = dice <= params_transform['flip_prob']

    if doflip:
        flip_order = params_transform['flip_order']
        img = img.copy()
        cv2.flip(src=img, flipCode=1, dst=img)
        w = img.shape[1]

        for i in range(len(meta['bodys'])):
            # change the coordinate
            meta['bodys'][i][:, 0] = w - 1 - meta['bodys'][i][:, 0]
            # change the left and the right
            meta['bodys'][i][:, :] = meta['bodys'][i][flip_order, :]

    return meta, img


def aug_rotate(meta, img, params_transform):
    dice = random.random()
    degree = (dice - 0.5) * 2 * \
        params_transform['max_rotate_degree']

    img_rot, R = rotate_bound(img, np.copy(degree), (128, 128, 128))

    # modify meta data
    for i in range(len(meta['bodys'])):
        meta['bodys'][i][:, :2] = rotate_skel2d(meta['bodys'][i][:, :2], R)

    return meta, img_rot


def rotate_bound(image, angle, bordervalue):
    """The correct way to rotation an image
       http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    """

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bordervalue), M


def rotate_skel2d(p2d, R):
    aug_p2d = np.concatenate((p2d, np.ones((p2d.shape[0], 1))), axis=1)
    rot_p2d = (R @ aug_p2d.T).T
    return rot_p2d[:, :2]



