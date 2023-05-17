import theseus as th

from .base_error import rmse, mean, r_t


def rpe_1(poses_gt: th.SE3, poses_est: th.SE3):
    return rmse(poses_gt, poses_est, 1)


def rpe_n(poses_gt: th.SE3, poses_est: th.SE3):
    rmp_n_sum = 0.0
    for distance in range(1, poses_gt.tensor.shape[0]):
        rmp_n_sum = rmp_n_sum + rmse(poses_gt, poses_est, distance)

    return rmp_n_sum / poses_gt.tensor.shape[0]


def l1_rpe_1(poses_gt: th.SE3, poses_est: th.SE3):
    return mean(poses_gt, poses_est, 1)


def l1_rpe_n(poses_gt: th.SE3, poses_est: th.SE3):
    rmp_n_sum = 0.0
    for distance in range(1, poses_gt.tensor.shape[0]):
        rmp_n_sum = rmp_n_sum + mean(poses_gt, poses_est, distance)

    return rmp_n_sum / poses_gt.tensor.shape[0]
