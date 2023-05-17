from math import cos, sin
from typing import List, Optional, Tuple, Union

import torch
import theseus as th


def skew(vector: torch.Tensor) -> torch.Tensor:
    r"""Return skew of a single/multple vectors
    The skew of a vector is defined by

    :math:`\hat{v} =`
    :math:`\bar{x}_k = \sum_{i=1}^{n_k}x_{ik}`

    Args:
        vector (torch.Tensor): vector or vectors of size ..., 3

    Returns:
        torch.Tensor: skew symmetric matrix/matrices of size ..., 3
    """

    skew_diag = torch.zeros_like(vector[..., 0])
    first_row = torch.stack([skew_diag, -vector[..., 2], vector[..., 1]], dim=-1)
    second_row = torch.stack([vector[..., 2], skew_diag, -vector[..., 0]], dim=-1)
    third_row = torch.stack([-vector[..., 1], vector[..., 0], skew_diag], dim=-1)

    return torch.stack([first_row, second_row, third_row], dim=-2)


def gaussian_kl_divergence(sigma_0: torch.Tensor, sigma_1: torch.Tensor) -> torch.Tensor:
    """Computes the KL-Divergence (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) for to zero mean mulitvariate gaussians
    KL = 1/2 * (log|simga_2| - log|sigma_1| - d + trace(sigma_2^-1 simga_1))
    https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians

    Args:
        sigma_0 (torch.Tensor): (..., N, N) covariance matrices of the first distribution
        sigma_1 (torch.Tensor): (..., N, N)covariance matrices of the second distribution

    Returns:
        float: KL-Divergence between the two gaussians
    """
    assert sigma_0.shape == sigma_1.shape
    assert sigma_0.shape[-1] == sigma_0.shape[-2]

    return 0.5 * (
        torch.log(torch.linalg.det(sigma_1) / torch.linalg.det(sigma_0))
        - sigma_0.shape[-1]
        + torch.einsum("...ij,...jk->...ik", torch.linalg.inv(sigma_1), sigma_0)
        .diagonal(offset=0, dim1=-2, dim2=-1)
        .sum(dim=-1)
    )


def sphere_to_carthesian(vector: torch.Tensor, jacobian: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if vector.shape[-1] != 2:
        raise ValueError(f"Expected the last dimension of vector to be of size 2, got {vector.shape[-1]} instead.")
    carthesian = torch.stack(
        [
            torch.cos(vector[..., 1]) * torch.sin(vector[..., 0]),
            torch.sin(vector[..., 1]) * torch.sin(vector[..., 0]),
            torch.cos(vector[..., 0]),
        ],
        dim=-1,
    )
    if not jacobian:
        return carthesian, None

    jac = torch.stack(
        [
            torch.stack(
                [
                    torch.cos(vector[..., 1]) * torch.cos(vector[..., 0]),
                    torch.sin(vector[..., 1]) * torch.cos(vector[..., 0]),
                    -torch.sin(vector[..., 0]),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    -torch.sin(vector[..., 1]) * torch.sin(vector[..., 0]),
                    torch.cos(vector[..., 1]) * torch.sin(vector[..., 0]),
                    torch.zeros_like(vector[..., 1]),
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )

    return carthesian, jac


def carthesian_to_sphere(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 3:
        raise ValueError(f"Expected the last dimension of vector to be of size 3, got {vector.shape[-1]} instead.")
    norm_vector = vector / torch.linalg.norm(vector, dim=-1)[..., None]
    return torch.stack(
        [torch.arccos(norm_vector[..., 2]), torch.arctan2(norm_vector[..., 1], norm_vector[..., 0])], dim=-1
    )


def rotation_matrix_2d(angle: Union[torch.Tensor, float]) -> torch.Tensor:
    if isinstance(angle, float):
        return torch.tensor([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    return torch.tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]])


Rotation = Union[torch.Tensor, th.SO3, th.SE3]


def angular_diff(rotation_0: Rotation, rotation_1: Rotation, degree: bool = True) -> torch.Tensor:
    """Returns the angular difference between two rotation matrices. The rotation matrices can be torch Tensors of shape that include a rotation matrix in the upper left corner, but also theseus SO3 and SE3 objects. Types can be mixed between variables.

    The angle is defined as

    \phi = arccos((R_1^\top R_2) - 1 / 2))

    Args:
        rotation_0 (Rotation): First rotation matrices
        rotation_1 (Rotation): Second rotation matrices
        degree (bool, optional): Return the angle in degree (True) or radian (False). Defaults to True.

    Returns:
        torch.Tensor: angular distance in degree or radian
    """

    def _to_so3(rotation: Rotation) -> th.SO3:
        if isinstance(rotation, torch.Tensor):
            if rotation.shape[-2] < 3 or rotation.shape[-1] < 3:
                raise TypeError(
                    f"Could not construct rotations from tensors as the dimensions are too small. Goten tensor of shape {rotation.shape}."
                )
            return th.SO3(tensor=rotation.flatten(-2, -1)[None, :3, :3])
        if isinstance(rotation, th.SE3):
            return th.SO3(tensor=rotation.tensor[..., :3, :3])
        return th.SO3(tensor=rotation.tensor[..., :3, :3])

    so3_0 = _to_so3(rotation_0)
    so3_1 = _to_so3(rotation_1)

    so3_0.to(torch.float32)
    so3_1.to(torch.float32)

    if so3_0.tensor.shape != so3_1.tensor.shape:
        raise TypeError(
            f"Did not receive the same number of rotations for evaluation. Got {so3_0.tensor.shape[0]} and {so3_1.tensor.shape[0]} rotations."
        )

    if degree:
        return torch.linalg.norm(so3_0.between(so3_1).log_map(), dim=1) * 180.0 / torch.pi
    else:
        return torch.linalg.norm(so3_0.between(so3_1).log_map(), dim=1)


Translation = Union[torch.Tensor, th.SE3]


def translational_diff(translations_0: Translation, translations_1: Translation, degree: bool = True) -> torch.Tensor:
    """Calculate the cosine error between the translations of two poses. The translations can either be given as torch tensors of theseus SE3 objects.

    Args:
        translations_0 (Translation): First translation matrices
        translations_1 (Translation): Second translation matrices.
        degree (bool, optional): Return the angle in degree (True) or radian (False). Defaults to True.

    Returns:
        torch.Tensor: cosine distance in degree or radian
    """

    def _to_tensor(translation: Translation) -> torch.Tensor:
        if isinstance(translation, th.SE3):
            return translation.tensor[..., :3, 3]
        return translation

    t_0 = _to_tensor(translations_0)
    t_1 = _to_tensor(translations_1)
    if t_0.shape != t_1.shape:
        raise TypeError(f"Did not receive the same translation shapes for evaluation. Got {t_0.shape} and {t_1.shape}.")

    cos_similarity = torch.nn.CosineSimilarity(dim=-1)
    pos_error = torch.arccos(torch.clip(cos_similarity(t_0, t_1), -1.0, 1.0))
    neg_error = torch.arccos(torch.clip(cos_similarity(t_0, -t_1), -1.0, 1.0))

    error = torch.min(torch.stack([pos_error, neg_error], dim=-1), dim=-1)[0]

    if degree:
        return error * 180.0 / torch.pi
    else:
        return error


def cycle_diff(poses_list: List[Union[th.SO3, th.SE3]], degree: bool = True) -> torch.Tensor:
    if isinstance(poses_list[0], th.SE3):
        so_3 = [th.SO3(tensor=poses.tensor[..., :3, :3]) for poses in poses_list]
    else:
        so_3 = poses_list

    rotations = so_3[0]
    for poses in so_3[1:]:
        rotations = rotations.compose(poses)
    if degree:
        return torch.linalg.norm(rotations.log_map(), dim=1) * 180.0 / torch.pi
    else:
        return torch.linalg.norm(rotations.log_map(), dim=1)


def make_positive_definite(matrices: torch.Tensor, epsilon: float = 1.0e-3) -> torch.Tensor:
    """Reconstruct 2x2 covariances, such that they satisfy the criterions to be a covariance matrix (be positive-definite and symmetric).
    The covariances passed to this function should already be symmetric. To fullfill the positive-definite criterion, Sylvester's criterions (https://en.wikipedia.org/wiki/Sylvester%27s_criterion) gives for a covariance matrix of the form:
          a b
    cov = b c,
    that a > 0, c > 0, b^2 < ac.

    Args:
        covariances (torch.Tensor): (..., 2, 2) covariance matrices that need to be corrected.
        epsilon (float): a, c must be at least this big

    Returns:
        torch.Tensor: (..., 2, 2) reconstructed covariance matrices that are positive-definite and symmetric
    """
    pd_matrices = torch.zeros_like(matrices)
    pd_matrices[..., 0, 0] = torch.clip(matrices[..., 0, 0], epsilon, None)
    pd_matrices[..., 1, 1] = torch.clip(matrices[..., 1, 1], epsilon, None)
    sqrt_ac = (
        torch.sqrt(torch.clip(matrices[..., 0, 0], epsilon, None) * torch.clip(matrices[..., 1, 1], epsilon, None))
        - epsilon
    )
    pd_matrices[..., 0, 1] = torch.clip((matrices[..., 0, 1] + matrices[..., 1, 0]) / 2.0, -sqrt_ac, sqrt_ac)
    pd_matrices[..., 1, 0] = torch.clip((matrices[..., 0, 1] + matrices[..., 1, 0]) / 2.0, -sqrt_ac, sqrt_ac)
    return pd_matrices
