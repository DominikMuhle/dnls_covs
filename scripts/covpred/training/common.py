import copy
from typing import Dict, List, Optional, Tuple

import torch
import theseus as th
from covpred.common import complete_pnec_covariances

from covpred.config.pose_estimation.config import PoseEstimationConfig
from covpred.math.common import skew
from covpred.math.pose_optimization import fast_denominator, numerator


def best_init_pose(
    init_poses: List[th.SE3],
    host_bvs: torch.Tensor,
    host_bvs_covs: Optional[torch.Tensor],
    target_bvs: torch.Tensor,
    target_bvs_covs: Optional[torch.Tensor],
    pose_estimation_config: PoseEstimationConfig,
) -> th.SE3:
    pnec = (host_bvs_covs is not None) or (target_bvs_covs is not None)
    if pnec:
        host_bvs_covs, target_bvs_covs = complete_pnec_covariances(host_bvs_covs, target_bvs_covs)
    # if host_bvs_covs is None:
    #     host_bvs_covs = torch.zeros_like(target_bvs_covs)
    # if target_bvs_covs is None:
    #     target_bvs_covs = torch.zeros_like(host_bvs_covs)

    def costs(poses: th.SE3) -> torch.Tensor:
        poses.to(host_bvs.dtype)
        if pnec:
            return (
                (
                    numerator(poses.tensor[..., :3, 3], skew(host_bvs), poses.tensor[..., :3, :3], target_bvs)
                    / fast_denominator(
                        poses.tensor[..., :3, 3],
                        poses.tensor[..., :3, :3],
                        skew(host_bvs),
                        host_bvs_covs,
                        target_bvs,
                        target_bvs_covs,
                        torch.ones_like(host_bvs)[..., 0] * pose_estimation_config.regularization,
                    )
                )
                .square()
                .sum(dim=-1)
            )
        else:
            return (
                numerator(poses.tensor[..., :3, 3], skew(host_bvs), poses.tensor[..., :3, :3], target_bvs)
                .square()
                .sum(dim=-1)
            )

    # best_poses = copy.deepcopy(init_poses[0].tensor)
    best_poses = init_poses[0].tensor.clone()
    best_poses = best_poses.to(torch.float64)
    best_costs = costs(init_poses[0])
    for poses in init_poses[1:]:
        new_costs = costs(poses)
        mask = new_costs < best_costs
        best_costs[mask] = new_costs[mask]
        best_poses[mask] = poses.tensor[mask]

    return th.SE3(tensor=best_poses)
