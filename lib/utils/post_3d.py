import numpy as np


def back_projection(x, d, K):
    """
    Back project 2D points x(2xN) to the camera coordinate and ignore distortion parameters.
    :param x: 2*N
    :param d: real depth of every point
    :param K: camera intrinsics
    :return: X (2xN), points in 3D
    """
    X = np.zeros((len(d), 3), np.float)
    X[:, 0] = (x[:, 0] - K[0, 2]) * d / K[0, 0]
    X[:, 1] = (x[:, 1] - K[1, 2]) * d / K[1, 1]
    X[:, 2] = d
    return X


def get_3d_points(pred_bodys, root_depth, K, root_n=2):
    bodys_3d = np.zeros(pred_bodys.shape, np.float)
    bodys_3d[:, :, 3] = pred_bodys[:, :, 3]
    for i in range(len(pred_bodys)):
        if pred_bodys[i][root_n][3] == 0:
            continue
        pred_bodys[i][:, 2] += root_depth[i]
        bodys_3d[i][:, :3] = back_projection(pred_bodys[i][:, :2], pred_bodys[i][:, 2], K)
    return bodys_3d