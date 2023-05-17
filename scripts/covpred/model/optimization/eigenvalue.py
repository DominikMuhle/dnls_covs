from typing import Tuple
from covpred.config.pose_estimation.config import NECConfig

import pyopengv
import theseus as th
import torch

from covpred.math.pose_optimization import gram_matrix, translation_from_gram_matrix


def nec(
    bvs_0: torch.Tensor,
    bvs_1: torch.Tensor,
    init_pose: th.SE3,
    config: NECConfig,
) -> Tuple[th.SE3, torch.Tensor]:
    init_rotation = init_pose.to_matrix()[0, :3, :3].to("cpu").detach().numpy()

    np_bvs_0 = bvs_0.to("cpu").detach().numpy()
    np_bvs_1 = bvs_1.to("cpu").detach().numpy()

    inlier_tensor = torch.zeros_like(bvs_0)[..., 0]
    if config.ransac:
        inliers = pyopengv.VecofInts()
        opengv_rotation = pyopengv.relative_pose_ransac_eigen(
            np_bvs_0, np_bvs_1, init_rotation, inliers, config.inlier_threshold
        )
        inlier_tensor[inliers] = 1.0
    else:
        opengv_rotation = pyopengv.relative_pose_eigensolver(np_bvs_0, np_bvs_1, init_rotation)
        inlier_tensor[:] = 1.0

    nec_rotation = torch.from_numpy(opengv_rotation).type_as(bvs_0)

    translation = init_pose.to_matrix()[0, :3, 3]
    if config.opt_t:
        translation = translation_from_gram_matrix(
            gram_matrix(
                bvs_0[inlier_tensor.to(torch.bool)][None], nec_rotation[None], bvs_1[inlier_tensor.to(torch.bool)][None]
            )
        )[0, ...]
    pose = torch.concat([nec_rotation, translation[:, None]], dim=-1)
    return th.SE3(tensor=pose[None, ...].type_as(init_pose.tensor)), inlier_tensor
