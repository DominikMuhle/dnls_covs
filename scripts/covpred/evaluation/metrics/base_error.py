# BSD 3-Clause License
#
# This file is part of the PNEC project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

from typing import List, Tuple
import theseus as th
import torch

from covpred.math.common import angular_diff


def rel_poses(poses: th.SE3, distance: int) -> th.SE3:
    host_poses = th.SE3(tensor=poses.tensor[:-distance, ...])
    target_poses = th.SE3(tensor=poses.tensor[distance:, ...])
    return host_poses.between(target_poses)


def error(poses_gt: th.SE3, poses_est: th.SE3, distance: int):
    # poses_gt: N
    # poses_est: N
    rel_gt_poses = rel_poses(poses_gt, distance)
    rel_est_poses = rel_poses(poses_est, distance)

    return angular_diff(rel_gt_poses, rel_est_poses)


def r_t(poses_gt: th.SE3, poses_est: th.SE3):
    rel_gt_poses = rel_poses(poses_gt, 1)
    gt_t = rel_gt_poses.tensor[..., :, 3]
    gt_norm = torch.linalg.norm(gt_t, dim=-1)
    rel_est_poses = rel_poses(poses_est, 1)
    est_t = rel_est_poses.tensor[..., :, 3]
    est_norm = torch.linalg.norm(est_t, dim=-1)

    pos_error = (
        torch.arccos(torch.clip(torch.einsum("...i,...i->...", gt_t, est_t) / (gt_norm * est_norm), -1.0, 1.0))
        * 180.0
        / torch.pi
    )
    neg_error = (
        torch.arccos(torch.clip(torch.einsum("...i,...i->...", gt_t, -est_t) / (gt_norm * est_norm), -1.0, 1.0))
        * 180.0
        / torch.pi
    )
    return torch.mean(torch.min(torch.stack([pos_error, neg_error], dim=-1), dim=-1)[0]).item()


def rmse(poses_gt: th.SE3, poses_est: th.SE3, distance: int):
    return torch.sqrt(torch.mean(torch.square(error(poses_gt, poses_est, distance)))).item()


def partial_rmse(poses_gt: th.SE3, poses_est: th.SE3, distance: int) -> Tuple[List[float], torch.Tensor]:
    errors = error(poses_gt, poses_est, distance)
    squared_errors = torch.square(errors)
    sort_errors = squared_errors.sort()[0]
    rmses = []
    for idx in range(sort_errors.shape[0]):
        rmses.append(torch.sqrt(torch.mean(sort_errors[: idx + 1])).item())
    return rmses, errors


def mean(poses_gt: th.SE3, poses_est: th.SE3, distance: int):
    return torch.mean(error(poses_gt, poses_est, distance)).item()
