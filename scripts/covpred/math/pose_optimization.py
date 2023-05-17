from enum import Enum
from typing import Optional, Tuple

import torch
from torch.cuda.amp.autocast_mode import custom_fwd
import theseus as th
from covpred.common import complete_pnec_covariances

from covpred.math.common import skew


# =================================================== EIGENVALUE NEC ===================================================
def gram_matrix(bvs_0: torch.Tensor, rotations: torch.Tensor, bvs_1: torch.Tensor) -> torch.Tensor:
    epipolar_normals = torch.einsum(
        "B...ij,Bjk,B...k->B...i",
        skew(bvs_0),
        rotations,
        bvs_1,
    )
    return torch.einsum("B...i,B...j->Bij", epipolar_normals, epipolar_normals)

@custom_fwd(cast_inputs=torch.float32)
def translation_from_gram_matrix(gram_matrices: torch.Tensor) -> torch.Tensor:
    eigenvectors = []
    if gram_matrices.ndim == 2:
        gram_matrices = gram_matrices[None]
    for gram_matrix in gram_matrices:
        # TODO: with amp error
        L, V = torch.linalg.eig(gram_matrix)
        V = V.real
        real_eigenvalues = L.real
        min_eigenvalue = real_eigenvalues[0]
        min_eigenvectors = V[:, 0] / torch.linalg.norm(V[:, 0])
        for idx, eigenvalue in enumerate(real_eigenvalues):
            if eigenvalue < min_eigenvalue:
                min_eigenvalue = eigenvalue
                min_eigenvectors = V[:, idx] / torch.linalg.norm(V[:, idx])
        eigenvectors.append(min_eigenvectors)
    return torch.stack(eigenvectors, dim=0)


# ====================================================== NUMERATOR =====================================================
def numerator(
    translation: torch.Tensor, bvs_0_hat: torch.Tensor, rotations: torch.Tensor, bvs_1: torch.Tensor
) -> torch.Tensor:
    return torch.einsum(
        "Bi,B...ij,Bjk,B...k->B...",
        translation,
        bvs_0_hat,
        rotations,
        bvs_1,
    )


def numerator_jac(
    translation: torch.Tensor, bvs_0_hat: torch.Tensor, rotations: torch.Tensor, bvs_1: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    jac_rot = torch.einsum(
        "B...i,Bji,B...jk->B...k",
        bvs_1,
        rotations,
        skew(torch.einsum("Bi,B...ij->B...j", translation, bvs_0_hat)),
    )

    jac_t = torch.einsum(
        "B...ij,Bjk,B...k->B...i",
        bvs_0_hat,
        rotations,
        bvs_1,
    )

    return jac_rot, jac_t


def epipolar_normal_covariance(
    rotations: torch.Tensor,
    bvs_0_hat: torch.Tensor,
    covariances_0: torch.Tensor,
    bvs_1: torch.Tensor,
    covariances_1: torch.Tensor,
) -> torch.Tensor:
    rot_bvs_1 = torch.einsum("Bij,B...j->B...i", rotations, bvs_1)
    return torch.einsum("B...ij,B...jk,B...lk->B...il", skew(rot_bvs_1), covariances_0, skew(rot_bvs_1)) + torch.einsum(
        "B...ij,Bjk,B...kl,Bml,B...nm->B...in",
        bvs_0_hat,
        rotations,
        covariances_1,
        rotations,
        bvs_0_hat,
    )


# ===================================================== DENOMINATOR ====================================================
def denominator(
    translation: torch.Tensor,
    rotations: torch.Tensor,
    bvs_0_hat: torch.Tensor,
    covariances_0: torch.Tensor,
    bvs_1: torch.Tensor,
    covariances_1: torch.Tensor,
    regularization: torch.Tensor,
) -> torch.Tensor:
    return torch.sqrt(
        torch.einsum(
            "Bi,B...ji,Bj->B...",
            translation,
            epipolar_normal_covariance(rotations, bvs_0_hat, covariances_0, bvs_1, covariances_1),
            translation,
        )
        + regularization
    )


def denominator_jac(
    translation: torch.Tensor,
    rotations: torch.Tensor,
    bvs_0_hat: torch.Tensor,
    covariances_0: torch.Tensor,
    bvs_1: torch.Tensor,
    covariances_1: torch.Tensor,
    regularization: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rot_bvs_1 = torch.einsum("Bij,B...j->B...i", rotations, bvs_1)
    sigmas = denominator(translation, rotations, bvs_0_hat, covariances_0, bvs_1, covariances_1, regularization)

    t_hat_sigma_t_hat_rot_bvs_1 = torch.einsum(
        "Bij,B...jk,B...lk,B...l->B...i",
        skew(translation),
        covariances_0,
        skew(translation),
        rot_bvs_1,
    )

    sigma_jac_rot = 2 * torch.einsum("B...i,B...ij->B...j", rot_bvs_1, skew(t_hat_sigma_t_hat_rot_bvs_1))

    sigma_prime_jac_rot = 2 * torch.einsum(
        "B...i,B...ij,Bjk,B...kl,Bml,B...mn->B...n",
        translation,
        bvs_0_hat,
        rotations,
        covariances_1,
        rotations,
        skew(torch.einsum("Bi,B...ij->B...j", translation, bvs_0_hat)),
    )

    rot_bvs_1 = torch.einsum("Bij,B...j->B...i", rotations, bvs_1)
    sigma_jac_t = 2 * torch.einsum(
        "B...ij,B...jk,B...lk,Bl->B...i",
        skew(rot_bvs_1),
        covariances_0,
        skew(rot_bvs_1),
        translation,
    )

    sigma_prime_jac_t = 2 * torch.einsum(
        "B...ij,Bjk,B...kl,Bml,B...nm,Bn->B...i",
        bvs_0_hat,
        rotations,
        covariances_1,
        rotations,
        bvs_0_hat,
        translation,
    )
    return (sigma_jac_rot + sigma_prime_jac_rot) / (2 * sigmas[..., None]), (sigma_jac_t + sigma_prime_jac_t) / (
        2 * sigmas[..., None]
    )


def epipolar_normal_covariance_0(
    rotations: torch.Tensor,
    covariances_0: torch.Tensor,
    bvs_1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rot_bvs_1 = torch.einsum("Bij,B...j->B...i", rotations, bvs_1)
    return torch.einsum("B...ij,B...jk,B...lk->B...il", skew(rot_bvs_1), covariances_0, skew(rot_bvs_1)), rot_bvs_1


def epipolar_normal_covariance_1(
    rotations: torch.Tensor,
    bvs_0_hat: torch.Tensor,
    covariances_1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bvs_0_hat_rot = torch.einsum("B...ij,Bjk->B...ik", bvs_0_hat, rotations)
    return torch.einsum("B...ij,B...jk,B...lk->B...il", bvs_0_hat_rot, covariances_1, bvs_0_hat_rot), bvs_0_hat_rot


def fast_denominator(
    translation: torch.Tensor,
    rotations: torch.Tensor,
    bvs_0_hat: torch.Tensor,
    covariances_0: torch.Tensor,
    bvs_1: torch.Tensor,
    covariances_1: torch.Tensor,
    regularization: torch.Tensor,
) -> torch.Tensor:
    return torch.sqrt(
        torch.einsum(
            "Bi,B...ji,B...j->B...",
            translation,
            epipolar_normal_covariance_0(rotations, covariances_0, bvs_1)[0]
            + epipolar_normal_covariance_1(rotations, bvs_0_hat, covariances_1)[0],
            translation,
        )
        + regularization
    )


def fast_denominator_jac(
    translation: torch.Tensor,
    rotations: torch.Tensor,
    bvs_0_hat: torch.Tensor,
    covariances_0: torch.Tensor,
    bvs_1: torch.Tensor,
    covariances_1: torch.Tensor,
    regularization: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rot_cov_0, rot_bvs_1 = epipolar_normal_covariance_0(rotations, covariances_0, bvs_1)
    rot_cov_1, bvs_0_hat_rot = epipolar_normal_covariance_1(rotations, bvs_0_hat, covariances_1)
    sigmas_times_2 = (
        2
        * torch.sqrt(
            torch.einsum(
                "Bi,B...ji,B...j->B...",
                translation,
                rot_cov_0 + rot_cov_1,
                translation,
            )
            + regularization
        )[..., None]
    )

    t_hat_sigma_t_hat_rot_bvs_1 = torch.einsum(
        "Bij,B...jk,B...lk,B...l->B...i",
        skew(translation),
        covariances_0,
        skew(translation),
        rot_bvs_1,
    )

    sigma_jac_rot = 2 * torch.einsum("B...i,B...ij->B...j", rot_bvs_1, skew(t_hat_sigma_t_hat_rot_bvs_1))

    sigma_prime_jac_rot = 2 * torch.einsum(
        "B...i,B...ik,B...kl,Bml,B...mn->B...n",
        translation,
        bvs_0_hat_rot,
        covariances_1,
        rotations,
        skew(torch.einsum("Bi,B...ij->B...j", translation, bvs_0_hat)),
    )

    sigma_jac_t = 2 * torch.einsum(
        "B...ij,Bj->B...i",
        rot_cov_0,
        translation,
    )

    sigma_prime_jac_t = 2 * torch.einsum(
        "B...ij,Bj->B...i",
        rot_cov_1,
        translation,
    )

    return (sigma_jac_rot + sigma_prime_jac_rot) / sigmas_times_2, (sigma_jac_t + sigma_prime_jac_t) / sigmas_times_2


class RPY(Enum):
    roll = 1
    pitch = 2
    yaw = 3


def RPYtoRotations(rolls: torch.Tensor, pitchs: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(rolls)
    ones = torch.ones_like(rolls)
    c_r, s_r = torch.cos(rolls), torch.sin(rolls)
    c_p, s_p = torch.cos(pitchs), torch.sin(pitchs)
    c_y, s_y = torch.cos(yaws), torch.sin(yaws)

    R_r = torch.stack([c_r, -s_r, zeros, s_r, c_r, zeros, zeros, zeros, ones], dim=-1).reshape(
        rolls.shape[0], rolls.shape[1], 3, 3
    )
    R_p = torch.stack([c_p, zeros, s_p, zeros, ones, zeros, -s_p, zeros, c_r], dim=-1).reshape(
        rolls.shape[0], rolls.shape[1], 3, 3
    )
    R_y = torch.stack([ones, zeros, zeros, zeros, c_y, -s_y, zeros, s_y, c_y], dim=-1).reshape(
        rolls.shape[0], rolls.shape[1], 3, 3
    )

    return torch.einsum("...ij,...jk,...kl->...il", R_r, R_p, R_y)


def evaluate_around_minimum(
    host_bvs: torch.Tensor,
    host_bvs_covs: Optional[torch.Tensor],
    target_bvs: torch.Tensor,
    target_bvs_covs: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
    gt_pose: th.SE3,
    regularization: float,
    axis: Tuple[RPY, RPY],
    x_ticks: torch.Tensor,
    y_ticks: torch.Tensor,
    degree=True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pnec = (host_bvs_covs is not None) or (target_bvs_covs is not None)
    if axis[0] == axis[1]:
        axis = (RPY.pitch, RPY.yaw)
        print(f"[WARNING]: You chose the same angle types. Switching to pitch and yaw!")
    if degree:
        x_ticks = x_ticks * torch.pi / 180.0
        y_ticks = y_ticks * torch.pi / 180.0
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks, indexing="ij")
    roll_grid = torch.zeros_like(grid_x)
    pitch_grid = torch.zeros_like(grid_x)
    yaw_grid = torch.zeros_like(grid_x)
    if axis[0] == RPY.roll:
        roll_grid = grid_x
    if axis[0] == RPY.pitch:
        pitch_grid = grid_x
    if axis[0] == RPY.yaw:
        yaw_grid = grid_x
    if axis[1] == RPY.roll:
        roll_grid = grid_y
    if axis[1] == RPY.pitch:
        pitch_grid = grid_y
    if axis[1] == RPY.yaw:
        yaw_grid = grid_y

    rotations = torch.einsum(
        "...ij,jk->...ik", RPYtoRotations(roll_grid, pitch_grid, yaw_grid), gt_pose.tensor[0, :3, :3]
    ).flatten(0, -3)

    num_rots = rotations.shape[0]
    cost = torch.sum(
        torch.square(
            numerator(
                gt_pose[..., :3, 3].expand(num_rots, -1),
                skew(host_bvs).expand(num_rots, -1, -1, -1),
                rotations,
                target_bvs.expand(num_rots, -1, -1),
            )
        ),
        dim=-1,
    ).reshape_as(
        grid_x
    )  # N,M

    if pnec:
        host_bvs_covs, target_bvs_covs = complete_pnec_covariances(host_bvs_covs, target_bvs_covs)
        # if host_bvs_covs is None:
        #     host_bvs_covs = torch.zeros_like(target_bvs_covs)
        # if target_bvs_covs is None:
        #     target_bvs_covs = torch.zeros_like(host_bvs_covs)

        cost = torch.sum(
            torch.square(
                numerator(
                    gt_pose[..., :3, 3].expand(num_rots, -1),
                    skew(host_bvs).expand(num_rots, -1, -1, -1),
                    rotations,
                    target_bvs.expand(num_rots, -1, -1),
                )
                / denominator(
                    gt_pose[0, :3, 3][None].expand(num_rots, -1),
                    rotations,
                    skew(host_bvs).expand(num_rots, -1, -1, -1),
                    host_bvs_covs.expand(num_rots, -1, -1, -1),
                    target_bvs.expand(num_rots, -1, -1),
                    target_bvs_covs.expand(num_rots, -1, -1, -1),
                    regularization=torch.ones((num_rots, 1)).type_as(host_bvs).to(host_bvs.device) * regularization,
                )
            ),
            dim=-1,
        ).reshape_as(
            grid_x
        )  # N,M
    if not pnec and weights is not None:
        cost = torch.sum(
            torch.square(
                numerator(
                    gt_pose[..., :3, 3].expand(num_rots, -1),
                    skew(host_bvs).expand(num_rots, -1, -1, -1),
                    rotations,
                    target_bvs.expand(num_rots, -1, -1),
                )
            )
            * weights[None],
            dim=-1,
        ).reshape_as(
            grid_x
        )  # N,M

    return grid_x, grid_y, cost
