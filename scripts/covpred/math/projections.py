from random import sample
from typing import Callable, Optional, Tuple, Literal, Union, overload
from numpy import cov

import torch

Projection = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor]]]


@overload
def linear(
    points: torch.Tensor,
    K_inv: torch.Tensor,
    covariances: None = None,
) -> torch.Tensor:
    ...


@overload
def linear(points: torch.Tensor, K_inv: torch.Tensor, covariances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


def linear(
    points: torch.Tensor,
    K_inv: torch.Tensor,
    covariances: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Does a linear projection from the image plane onto the unit sphere
    bearing vector, using
    p_3d = K_inv * point
    bv = p_3d / ||p_3d||
    \Sigma_{bv} = J K_inv \Sigma K_inv^\top J^\top

    J = (||p_3d||^2 * I - p_3d * P-3d^\top) / ||p_3d||^3

    Args:
        point (torch.Tensor): image point in homogenous coordinates (...xNx3)
        K_inv (torch.Tensor): inverse camera matrix (...x3x3)
        covariance (Optional[torch.Tensor]): covariances corresponding to the points (...xNx3x3), default = None

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: projected points (...xNx3) and covariance matrices (...xNx3x3) (if given)
    """

    transformed_points = torch.einsum("...ij,...nj->...ni", K_inv, points)
    norms = torch.norm(transformed_points, dim=-1)

    if covariances is not None:
        jacobians = (1 / torch.pow(norms, 3))[..., None, None] * (
            torch.pow(norms, 2)[..., None, None] * torch.eye(3, device=transformed_points.device, dtype=points.dtype)[None, ...]
            - torch.einsum("...i,...j->...ij", transformed_points, transformed_points)
        )
        covariances = torch.einsum(
            "...nij,...jk,...nkl,...ml,...npm->...nip",
            jacobians,
            K_inv,
            covariances,
            K_inv,
            jacobians,
        )
        transformed_points = transformed_points / norms[..., None]
        return transformed_points, covariances

    transformed_points = transformed_points / norms[..., None]
    return transformed_points


def unscented(
    kappa: float = 1.0,
    unscented_mean: bool = True,
) -> Projection:
    """Configures unscented projection for further use

    Args:
        kappa (float, optional): kappa parameter for the unscented projection. Defaults to 1.0.
        unscented_mean (bool, optional): use the unscented mean as mean. Defaults to True.

    Returns:
        Projection: Callable that does the unscented transformation
    """

    def _unscented(
        points: torch.Tensor,
        K_inv: torch.Tensor,
        covariances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perfoms the unscented transformation of a covariance from the image plane of a pinhole camera onto the unit sphere in 3D. Accounts for the rank deficiency of the covariance (max rank = 2). Defaults to linear projection if not covariances are provided.

        Args:
            point (torch.Tensor): image point in homogenous coordinates (...xNx3)
            K_inv (torch.Tensor): inverse camera matrix (...x3x3)
            covariance (Optional[torch.Tensor]): covariances corresponding to the points (...xNx3x3), default = None

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: projected points (...xNx3) and covariance matrices (...xNx3x3) (if given)
        """
        if covariances is None:
            return linear(points, K_inv, covariances)

        n = 2

        chol_covs = torch.linalg.cholesky(covariances[..., :2, :2])

        weights = torch.zeros((2 * n + 1), dtype=torch.float64)
        sample_points = []
        weights[0] = kappa / (n + kappa)
        sample_points.append(points)
        for i in range(n):
            sample_points.append(points + torch.nn.functional.pad(chol_covs[..., :, i], (0, 1), value=0.0))
            weights[i + 1] = 0.5 / (n + kappa)
            sample_points.append(points - torch.nn.functional.pad(chol_covs[..., :, i], (0, 1), value=0.0))
            weights[i + 1 + n] = 0.5 / (n + kappa)
        sample_points = torch.stack(sample_points, dim=-1)

        transformed_points = torch.einsum("ij,...jk->...ik", K_inv, sample_points)
        transformed_points = transformed_points / torch.norm(transformed_points, dim=-2)[..., None, :]

        means = torch.sum(weights * transformed_points, dim=-1)

        difference = transformed_points - means[..., :, None]

        covs = torch.einsum("...ij,j,...kj->...ik", difference, weights, difference)

        if not unscented_mean:
            # use the projected point as the mean.
            means = transformed_points[:, 0]

        return means, covs

    return _unscented
