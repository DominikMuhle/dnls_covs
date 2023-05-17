import logging
from typing import Optional, Tuple
from matplotlib.colors import hsv_to_rgb

import numpy as np
import torch

from covpred.math.common import make_positive_definite


def covs_from_parameters(
    parameters: torch.Tensor, positive_definite: bool = False, isotropic: bool = False
) -> torch.Tensor:
    """Return covariances constructed using scale (s), orientation (alpha)
    and anisotropy level (beta) parameters using.
    cov =  s * R * Sigma * R.T
    with
            cos(alpha) -sin(alpha)
    R   =   sin(alpha)  cos(alpha)

            1 - beta    0
    T   =   0           beta

    Args:
        parameters (torch.Tensor): array of size [..., 3] containing scale, rotation, anisotropy

    Returns:
        torch.Tensor: array of size [...x 2 x 2]
    """
    s = parameters[..., 0]
    if isotropic:
        return torch.stack(
            [torch.stack([s, torch.zeros_like(s)], dim=-1), torch.stack([torch.zeros_like(s), s], dim=-1)], dim=-1
        )

    alpha = parameters[..., 1]
    beta = parameters[..., 2]
    assert s.shape == alpha.shape and s.shape == beta.shape

    c = torch.diag_embed(torch.stack([1 - beta, beta], dim=-1))
    rot = torch.stack(
        [torch.cos(alpha), -torch.sin(alpha), torch.sin(alpha), torch.cos(alpha)],
        dim=-1,
    ).reshape(
        alpha.shape
        + (
            2,
            2,
        )
    )

    reconst_covs = s[..., None, None] * torch.einsum("...ij,...jk,...lk->...il", rot, c, rot)

    if positive_definite:
        return make_positive_definite(reconst_covs)
    return reconst_covs


def parameters_from_covs(covs: torch.Tensor, scale_limits: Tuple[Optional[float], Optional[float]]) -> torch.Tensor:
    """Return the scale (s), orientation (alpha) and anisotropy level (beta) of
    given [2 x 2] covariances. The single covariances can be reconstructed using
    cov =  s * R * Sigma * R.T
    with
            cos(alpha) -sin(alpha)
    R   =   sin(alpha)  cos(alpha)

            1 - beta    0
    T   =   0           beta

    Args:
        vector (torch.Tensor): array of size [...x 2 x 2]

    Returns:
        torch.Tensor: array of size [...x 3]
    """
    assert covs.shape[-2:] == (2, 2)

    # scaling
    s = covs[..., 0, 0] + covs[..., 1, 1]

    # orientation
    alpha = torch.zeros(covs.shape[:-2], dtype=covs.dtype, device=covs.device)
    alpha_1 = torch.zeros(covs.shape[:-2], dtype=covs.dtype, device=covs.device)
    alpha_2 = torch.zeros(covs.shape[:-2], dtype=covs.dtype, device=covs.device)
    non_zero = covs[..., 0, 1] != 0.0
    b = (covs[..., 0, 0][non_zero] - covs[..., 1, 1][non_zero]) / covs[..., 0, 1][non_zero]
    alpha_1[non_zero] = torch.arctan(-(b / 2) - torch.sqrt(torch.square(b / 2) + 1))
    alpha_2[non_zero] = torch.arctan(-(b / 2) + torch.sqrt(torch.square(b / 2) + 1))

    alpha[covs[..., 0, 1] >= 0] = alpha_1[covs[..., 0, 1] >= 0]
    alpha[covs[..., 0, 1] < 0] = alpha_2[covs[..., 0, 1] < 0]

    rot = torch.stack(
        [torch.cos(alpha), -torch.sin(alpha), torch.sin(alpha), torch.cos(alpha)],
        dim=-1,
    ).reshape(
        alpha.shape
        + (
            2,
            2,
        )
    )

    base_cov = torch.einsum("...ji,...jk,...kl->...il", rot, covs, rot) / s[..., None, None]

    beta = base_cov[..., 1, 1]

    if scale_limits[0] is not None or scale_limits[1] is not None:
        s = torch.clip(s, scale_limits[0], scale_limits[1])

    return torch.stack([s, alpha, beta], dim=-1)


def entries_from_covs(covariances: torch.Tensor) -> torch.Tensor:
    """Returns the unrolled upper triangle from covariance matrices.
          a b
    Cov = b c
    Returns a b c

    Args:
        covariances (torch.Tensor): (..., 2, 2) covariance matrices

    Returns:
        torch.Tensor: (..., 3) unrolled upper triangle from covariance matrices
    """
    assert covariances.shape[-1] == covariances.shape[-2]
    assert covariances.shape[-1] == 2

    # unroll the last two dimensions of the array
    covariances = covariances.reshape(*covariances.shape[:-2], 4)
    return covariances[..., [0, 1, 3]]


def covs_from_entries(entries: torch.Tensor, positive_definite: bool = False, isotropic: bool = False) -> torch.Tensor:
    """Reconstruct covariance matrix from its entries.
    Entries = a b c
            a b
    Returns b c

    Args:
        entries (torch.Tensor): (..., 3) entries for covariance matrices

    Returns:
        torch.Tensor: (..., 2, 2) covariance matrices
    """
    assert entries.shape[-1] == 3
    if isotropic:
        return torch.stack(
            [
                torch.stack([entries[..., 0], torch.zeros_like(entries[..., 0])], dim=-1),
                torch.stack([torch.zeros_like(entries[..., 0]), entries[..., 0]], dim=-1),
            ],
            dim=-1,
        )
    covariances = entries[..., [0, 1, 1, 2]]
    if positive_definite:
        return make_positive_definite(covariances.reshape(*covariances.shape[:-1], 2, 2))
    return covariances.reshape(*covariances.shape[:-1], 2, 2)


def entries_from_inv_covs(covariances: torch.Tensor) -> torch.Tensor:
    """Returns the unrolled upper triangle from covariance matrices inverses.
          a b
    Cov = b c
    Returns 1/det(Cov) c -b a

    Args:
        covariances (torch.Tensor): (..., 2, 2) covariance matrices

    Returns:
        torch.Tensor: (..., 3) unrolled upper triangle from covariance matrices
        inverses
    """
    assert covariances.shape[-1] == covariances.shape[-2]
    assert covariances.shape[-1] == 2

    inv_covariances = torch.linalg.inv(covariances)
    return entries_from_covs(inv_covariances)


def covs_from_inverse_entries(
    entries: torch.Tensor, positive_definite: bool = False, isotropic: bool = False
) -> torch.Tensor:
    """Reconstruct covariance matrix from the entries of its inverse.
    Entries = a b c
                    c -b
    Returns ab-bc * -b a

    Args:
        entries (torch.Tensor): (..., 3) entries for covariance matrices

    Returns:
        torch.Tensor: (..., 2, 2) covariance matrices
    """
    assert entries.shape[-1] == 3
    if isotropic:
        return torch.stack(
            [
                torch.stack([1 / entries[..., 0], torch.zeros_like(entries[..., 0])], dim=-1),
                torch.stack([torch.zeros_like(entries[..., 0]), 1 / entries[..., 0]], dim=-1),
            ],
            dim=-1,
        )
    inv_covariances = entries[..., [0, 1, 1, 2]]
    inv_covariances = inv_covariances.reshape(*inv_covariances.shape[:-1], 2, 2)
    if positive_definite:
        return make_positive_definite(torch.linalg.inv(inv_covariances))
    return torch.linalg.inv(inv_covariances)


def rot_eigen_from_sab(parameters: torch.Tensor) -> torch.Tensor:
    """Return orientation (alpha) and eigenvalues (lambda_1, lambda_2) constructed using scale (s), orientation (alpha)
    and anisotropy level (beta) parameters using.
    cov =  R * Sigma * R.T
    with
              cos(alpha) -sin(alpha)
    R     =   sin(alpha)  cos(alpha)

              lambda_1    0
    Sigma =   0           lambda_2

    Args:
        parameters (torch.Tensor): array of size [..., 3] containing scale, rotation, anisotropy

    Returns:
        torch.Tensor: array of size [...x 2 x 2]
    """
    s = parameters[..., 0]
    alpha = parameters[..., 1]
    beta = parameters[..., 2]
    lambda_1 = s * (1 - beta)
    lambda_2 = s * beta
    return torch.stack([alpha, lambda_1, lambda_2], dim=-1)


def sab_from_rot_eigen(parameters: torch.Tensor) -> torch.Tensor:
    """Return orientation (alpha) and eigenvalues (lambda_1, lambda_2) constructed using scale (s), orientation (alpha)
    and anisotropy level (beta) parameters using.
    cov =  R * Sigma * R.T
    with
              cos(alpha) -sin(alpha)
    R     =   sin(alpha)  cos(alpha)

              lambda_1    0
    Sigma =   0           lambda_2

    Args:
        parameters (torch.Tensor): array of size [..., 3] containing scale, rotation, anisotropy

    Returns:
        torch.Tensor: array of size [...x 2 x 2]
    """
    alpha = parameters[..., 0]
    lambda_1 = parameters[..., 1]
    lambda_2 = parameters[..., 2]
    s = lambda_1 + lambda_2
    beta = lambda_2 / s
    return torch.stack([s, alpha, beta], dim=-1)


visusalization_logger = logging.getLogger("visualization")


def sab_to_image(
    parameters: torch.Tensor, scale_limit: float | None = None, gamma: float = 1.0, beta_gamma: float = 1.0
) -> np.ndarray:
    scale = parameters[..., :, :, 0]
    if scale_limit is None and scale.max() > 1.0:
        visusalization_logger.warning(
            f"Covariances were not normalized for Visualization. Normalizing with current maximum scale of {scale.max()}"
        )
        visusalization_logger.warning(f"No consitent visualization of covariances can be guaranteed.")
        scale = scale / scale.max()

    if scale_limit is not None:
        scale = torch.clip(scale, max=scale_limit) / scale_limit

    # gamma correction
    scale = torch.pow(scale, gamma)
    angle = parameters[..., :, :, 1]
    angle = torch.remainder(angle, torch.pi) / torch.pi

    beta = parameters[..., :, :, 2]
    beta[beta < 0.5] = 1 - beta[beta < 0.5]
    beta = (beta - 0.5) * 2.0
    beta = torch.pow(beta, beta_gamma)

    return hsv_to_rgb(torch.stack([angle, beta, scale], dim=-1).detach().to("cpu").numpy())
    # return hsv_to_rgb(torch.stack([beta, angle,  scale], dim=-1).detach().to("cpu").numpy())


def entries_to_image(entries: torch.Tensor, scale_limits: Tuple[float | None, float | None]) -> np.ndarray:
    return sab_to_image(parameters_from_covs(covs_from_entries(entries), scale_limits), scale_limits[1])


def inverse_entries_to_image(
    inverse_entries: torch.Tensor, scale_limits: Tuple[float | None, float | None]
) -> np.ndarray:
    return sab_to_image(parameters_from_covs(covs_from_inverse_entries(inverse_entries), scale_limits), scale_limits[1])


def covs_to_image(covariances: torch.Tensor, scale_limits: Tuple[float | None, float | None]) -> np.ndarray:
    return sab_to_image(parameters_from_covs(covariances, scale_limits), scale_limits[1], 0.9, 0.8)
