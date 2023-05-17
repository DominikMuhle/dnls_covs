import itertools
from typing import Dict, List

import torch
import theseus as th

from covpred.common import to_3d_cov, to_3d_point
from covpred.math.projections import linear
from covpred.math.common import skew, gaussian_kl_divergence
from covpred.math.pose_optimization import denominator


def sigma_2_error(sigma_2: List[torch.Tensor], normalize: bool = False) -> torch.Tensor:
    N = len(sigma_2)
    errors = torch.zeros(N, N, sigma_2[0].shape[0])
    for i in range(N):
        sigma_i = sigma_2[i]
        if normalize:
            sigma_i = sigma_i / torch.mean(sigma_i, dim=-1, keepdim=True)
        for j in range(N):
            sigma_j = sigma_2[j]
            if normalize:
                sigma_j = sigma_j / torch.mean(sigma_j, dim=-1, keepdim=True)
            errors[i, j] = torch.abs(sigma_i - sigma_j).mean(dim=-1)
    return errors


def image_covariance_error(covariances: List[torch.Tensor], normalize: bool = False) -> torch.Tensor:
    N = len(covariances)
    errors = torch.zeros(N, N, covariances[0].shape[0])
    for i in range(N):
        Sigma_i = covariances[i]
        if normalize:
            Sigma_i = Sigma_i / torch.mean(torch.trace(Sigma_i), dim=-1, keepdim=True)
        for j in range(N):
            Sigma_j = covariances[j]
            if normalize:
                Sigma_j = Sigma_j / torch.mean(torch.trace(Sigma_j), dim=-1, keepdim=True)
            errors[i, j] = gaussian_kl_divergence(Sigma_i, Sigma_j)

    return errors


def covariance_similarity(
    host_points: torch.Tensor,
    target_points: torch.Tensor,
    host_covariances: List[torch.Tensor],
    target_covariances: List[torch.Tensor],
    K_inv: torch.Tensor,
    pose: th.SE3,
    regularization: float,
) -> Dict[str, torch.Tensor]:
    device = host_points.device
    num_problems = host_points.shape[0]
    host_bvs = linear(to_3d_point(host_points), K_inv[None].expand(num_problems, -1, -1))
    target_bvs = linear(to_3d_point(target_points), K_inv[None].expand(num_problems, -1, -1))
    host_bvs_covs: List[torch.Tensor] = []
    target_bvs_covs: List[torch.Tensor] = []
    sigma_2: List[torch.Tensor] = []

    for host_covs, target_covs in zip(host_covariances, target_covariances):
        host_projection = linear(
            to_3d_point(host_points),
            K_inv[None].expand(num_problems, -1, -1),
            to_3d_cov(host_covs[None].expand(num_problems, -1, -1, -1)),
        )[1]
        host_bvs_covs.append(host_projection)
        target_projection = linear(
            to_3d_point(host_points),
            K_inv[None].expand(num_problems, -1, -1),
            to_3d_cov(target_covs[None].expand(num_problems, -1, -1, -1)),
        )[1]
        target_bvs_covs.append(target_projection)

        sigma_2.append(
            denominator(
                pose.tensor[:, :3, 3].to(device),
                pose.tensor[:, :3, :3].to(device),
                skew(host_bvs),
                host_projection,
                target_bvs,
                target_projection,
                torch.ones(num_problems, 1, device=device) * regularization,
            )
        )

    return {
        "sigma_2": sigma_2_error(sigma_2),
        "host_kl_divergence": image_covariance_error(host_covariances),
        "target_kl_divergence": image_covariance_error(target_covariances),
    }
