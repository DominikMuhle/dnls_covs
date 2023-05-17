from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from covpred.math.covariance_filter import CovarianceFilter


def padded_img_and_grad(img: torch.Tensor, padding: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    padded_img = torch.nn.functional.pad(img, (padding + 1, padding + 1, padding + 1, padding + 1), value=0.0)

    gradient_x = padded_img[..., :-1, 1:] - padded_img[..., :-1, :-1]
    gradient_y = padded_img[..., 1:, :-1] - padded_img[..., :-1, :-1]

    return padded_img[..., :-1, :-1], torch.stack([gradient_x, gradient_y], dim=-1)


patterns: Dict[str, List[Tuple[int, int]]] = {
    "pattern52": [
        (-3, 7),
        (-1, 7),
        (1, 7),
        (3, 7),
        (-5, 5),
        (-3, 5),
        (-1, 5),
        (1, 5),
        (3, 5),
        (5, 5),
        (-7, 3),
        (-5, 3),
        (-3, 3),
        (-1, 3),
        (1, 3),
        (3, 3),
        (5, 3),
        (7, 3),
        (-7, 1),
        (-5, 1),
        (-3, 1),
        (-1, 1),
        (1, 1),
        (3, 1),
        (5, 1),
        (7, 1),
        (-7, -1),
        (-5, -1),
        (-3, -1),
        (-1, -1),
        (1, -1),
        (3, -1),
        (5, -1),
        (7, -1),
        (-7, -3),
        (-5, -3),
        (-3, -3),
        (-1, -3),
        (1, -3),
        (3, -3),
        (5, -3),
        (7, -3),
        (-5, -5),
        (-3, -5),
        (-1, -5),
        (1, -5),
        (3, -5),
        (5, -5),
        (-3, -7),
        (-1, -7),
        (1, -7),
        (3, -7),
    ],
    "pattern24": [
        (-1, 5),
        (1, 5),
        (-3, 3),
        (-1, 3),
        (1, 3),
        (3, 3),
        (-5, 1),
        (-3, 1),
        (-1, 1),
        (1, 1),
        (3, 1),
        (5, 1),
        (-5, -1),
        (-3, -1),
        (-1, -1),
        (1, -1),
        (3, -1),
        (5, -1),
        (-3, -3),
        (-1, -3),
        (1, -3),
        (3, -3),
        (-1, -5),
        (1, -5),
    ],
}


def _unsc_impl(img: torch.Tensor, pattern_name: str = "pattern52", precision: torch.dtype = torch.float32):
    # with torch.no_grad():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    img = img.to(precision)

    H = img.shape[-2]  # y-direction
    W = img.shape[-1]  # x-direction
    pattern = patterns[pattern_name]
    pattern_size = len(pattern)
    padding = pattern[0][1]
    padded_img, grad = padded_img_and_grad(img, padding)

    J_se2 = torch.concat(
        [
            torch.eye(2, device=img.device, dtype=precision)[None, ...].expand(pattern_size, -1, -1),  # m,2,2
            torch.stack([torch.tensor([-y, x], device=img.device, dtype=precision) for x, y in pattern], dim=0)[
                ..., None
            ],  # m,2,1
        ],
        dim=-1,
    )  # m,2,3

    intensity_sum = torch.zeros_like(img)  # ...,H,W
    grad_se2_sum = torch.stack(3 * [torch.zeros_like(img)], dim=-1)  # ...,H,W,3
    for idx, (x, y) in enumerate(pattern):
        intensity_sum = (
            intensity_sum
            + padded_img[
                ...,
                (y + padding) : H + (y + padding),
                (x + padding) : W + (x + padding),
            ]
        )
        grad_se2_sum = grad_se2_sum + torch.einsum(
            "...HWi,ij->...HWj",
            grad[
                ...,
                (y + padding) : H + (y + padding),
                (x + padding) : W + (x + padding),
                :,
            ],
            J_se2[idx, ...],
        )

    H_se2 = torch.stack(3 * [torch.stack(3 * [torch.zeros_like(img)], dim=-1)], dim=-1)  # ...H,W,3,3
    for idx, (x, y) in enumerate(pattern):
        J_pi = (
            pattern_size
            * (
                torch.einsum(
                    "...HWi,ij->...HWj",
                    grad[
                        ...,
                        (y + padding) : H + (y + padding),
                        (x + padding) : W + (x + padding),
                        :,
                    ],
                    J_se2[idx, ...],
                )
                * intensity_sum[..., None]
                - padded_img[
                    ...,
                    (y + padding) : H + (y + padding),
                    (x + padding) : W + (x + padding),
                ][..., None]
                * grad_se2_sum
            )
            / (intensity_sum[..., None] * intensity_sum[..., None])
        )
        H_se2 = H_se2 + torch.einsum("...HWi,...HWj->...HWij", J_pi, J_pi)

    Hse_2_dampend = H_se2 + 1.0e-3 * torch.eye(3, device=H_se2.device)[None, None, None, None, :, :]
    H_se2_inv = (
        torch.stack(
            [
                Hse_2_dampend[..., 1, 1],
                -Hse_2_dampend[..., 0, 1],
                -Hse_2_dampend[..., 0, 1],
                Hse_2_dampend[..., 0, 0],
            ],
            dim=-1,
        )
        / (
            (Hse_2_dampend[..., 1, 1] * Hse_2_dampend[..., 0, 0])
            - (Hse_2_dampend[..., 0, 1] * Hse_2_dampend[..., 1, 0])
        )[..., None]
    ).reshape(
        Hse_2_dampend.shape[:-2]
        + (
            2,
            2,
        )
    )
    end.record()
    torch.cuda.synchronize()
    # print(f"Total: {start.elapsed_time(end)}")
    # invert off-diagonal elements due to indexing of matrices
    H_se2_inv[..., 1, 0] = H_se2_inv[..., 0, 1] = -H_se2_inv[..., 1, 0]
    return H_se2_inv


def _unsc_impl_fast(
    img: torch.Tensor, pattern_name: str = "pattern52", precision: torch.dtype = torch.float32
) -> torch.Tensor:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    img = img.to(precision)
    H = img.shape[-2]  # y-direction
    W = img.shape[-1]  # x-direction
    pattern = patterns[pattern_name]
    pattern_size = len(pattern)
    padding = pattern[0][1]
    padded_img, grad = padded_img_and_grad(img, padding)
    grad_x = grad[..., 0]
    grad_y = grad[..., 1]

    intensity_sum = torch.zeros_like(img)  # ...,H,W
    intensity_squared_sum = torch.zeros_like(img)  # ...,H,W
    grad_se2_sum = torch.stack(3 * [torch.zeros_like(img)], dim=-1)  # ...,H,W,3
    weighted_grad_se2_sum = torch.stack(3 * [torch.zeros_like(img)], dim=-1)  # ...,H,W,3
    grad_se2_T_grad_se2_sum = torch.stack(3 * [torch.stack(3 * [torch.zeros_like(img)], dim=-1)], dim=-1)  # ...H,W,3,3

    for idx, (x, y) in enumerate(pattern):
        intensity = padded_img[
            ...,
            (y + padding) : H + (y + padding),
            (x + padding) : W + (x + padding),
        ]
        grad_x_ = grad_x[
            ...,
            (y + padding) : H + (y + padding),
            (x + padding) : W + (x + padding),
        ]
        grad_y_ = grad_y[
            ...,
            (y + padding) : H + (y + padding),
            (x + padding) : W + (x + padding),
        ]
        grad_se2 = torch.stack([grad_x_, grad_y_, x * grad_y_ - y * grad_x_], dim=-1)
        intensity_sum = intensity_sum + intensity
        intensity_squared_sum = intensity_squared_sum + (intensity * intensity)
        grad_se2_sum = grad_se2_sum + grad_se2
        weighted_grad_se2_sum = weighted_grad_se2_sum + (intensity[..., None] * grad_se2)
        grad_se2_T_grad_se2_sum = grad_se2_T_grad_se2_sum + torch.einsum("...i,...j->...ij", grad_se2, grad_se2)
    end.record()
    torch.cuda.synchronize()
    # print(f"Total: {start.elapsed_time(end)}")

    start.record()

    weighted_grad_se2_sum_grad_se2_sum = torch.einsum("...i,...j->...ij", weighted_grad_se2_sum, grad_se2_sum)
    intensity_sum_squared = intensity_sum * intensity_sum
    H_se2 = (
        (intensity_sum_squared)[..., None, None] * grad_se2_T_grad_se2_sum
        - intensity_sum[..., None, None]
        * (weighted_grad_se2_sum_grad_se2_sum + weighted_grad_se2_sum_grad_se2_sum.transpose(-1, -2))
        + intensity_squared_sum[..., None, None] * torch.einsum("...i,...j->...ij", grad_se2_sum, grad_se2_sum)
    ) / (intensity_sum_squared * intensity_sum_squared)[..., None, None]

    Hse_2_dampend = H_se2 + 1.0e-3 * torch.eye(3, device=H_se2.device)[None, None, None, None, :, :]
    H_se2_inv = (
        torch.stack(
            [
                Hse_2_dampend[..., 1, 1],
                -Hse_2_dampend[..., 0, 1],
                -Hse_2_dampend[..., 0, 1],
                Hse_2_dampend[..., 0, 0],
            ],
            dim=-1,
        )
        / (
            (Hse_2_dampend[..., 1, 1] * Hse_2_dampend[..., 0, 0])
            - (Hse_2_dampend[..., 0, 1] * Hse_2_dampend[..., 1, 0])
        )[..., None]
    ).reshape(
        Hse_2_dampend.shape[:-2]
        + (
            2,
            2,
        )
    )
    end.record()
    torch.cuda.synchronize()
    # print(f"Combination: {start.elapsed_time(end)}")

    H_se2_inv[..., 1, 0] = H_se2_inv[..., 0, 1] = -H_se2_inv[..., 1, 0]
    return H_se2_inv


def get_klt_unsc(
    img: torch.Tensor,
    pattern_name: str = "pattern52",
    precision: torch.dtype = torch.float32,
    scaling: float = 1.0,
    filter_function: Optional[CovarianceFilter] = None,
) -> torch.Tensor:
    unsc = _unsc_impl(img, pattern_name, precision)
    if filter_function is not None:
        unsc = filter_function(unsc)
    # fast = _unsc_impl_fast(img, pattern_name, precision)
    return unsc * scaling


def interp_intensity_and_grad(patch: np.ndarray, img_feature_position: Tuple) -> np.ndarray:
    """Interpolate the patch intensity and gradient based on the subpixel
    feature position. The patch has the size (N+3, N+3), where (N, N) is the
    region of interest.

    Args:
        patch (np.ndarray): square numpy array containing the patch of
                            size (N+3, N+3)
        img_feature_position (tuple): subpixel feature position in the image

    Returns:
        np.ndarray: array of size (N, N, 3) containing the interpolated intesity
                    and gradient
    """
    assert patch.ndim == 2
    assert patch.shape[0] == patch.shape[1]

    N = patch.shape[0] - 3
    dx = img_feature_position[0] - int(img_feature_position[0])
    dy = img_feature_position[1] - int(img_feature_position[1])
    ddx = 1.0 - dx
    ddy = 1.0 - dy

    intesity = np.zeros((N, N))
    grad = np.zeros((N, N, 2))
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            px0y0 = patch[x, y]
            px1y0 = patch[x + 1, y]
            px0y1 = patch[x, y + 1]
            px1y1 = patch[x + 1, y + 1]

            interp_intesity = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 + dx * dy * px1y1
            intesity[x - 1, y - 1] = interp_intesity

            pxm1y0 = patch[x - 1, y]
            pxm1y1 = patch[x - 1, y + 1]

            res_mx = ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 + dx * dy * px0y1

            px2y0 = patch[x + 2, y]
            px2y1 = patch[x + 2, y + 1]

            res_px = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 + dx * dy * px2y1

            grad[x - 1, y - 1, 0] = 0.5 * (res_px - res_mx)

            px0ym1 = patch[x, y - 1]
            px1ym1 = patch[x + 1, y - 1]

            res_my = ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 + dx * dy * px1y0

            px0y2 = patch[x, y + 2]
            px1y2 = patch[x + 1, y + 2]

            res_py = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 + dx * dy * px1y2

            grad[x - 1, y - 1, 1] = 0.5 * (res_py - res_my)

    return intesity, grad


def inv_comp_uncertainty(patch: np.ndarray, img_feature_position: tuple) -> np.ndarray:
    """Computes the uncertainty of a feature position based on a surrounding
    patch of size (N + 3, N + 3).

    Args:
        patch (np.ndarray): square numpy array containing the patch of
                            size (N+3, N+3)
        img_feature_position (tuple): subpixel feature position in the image

    Returns:
        np.ndarray: covariance matrix for the feature position uncertainty as
                    a (2, 2) matrix
    """
    N = patch.shape[0] - 3
    # (N, N, 3)
    intesity, grad = interp_intensity_and_grad(patch, img_feature_position)

    intesity_sum = np.sum(intesity)
    grad_sum_se2 = np.zeros((1, 3))
    J_se2 = np.zeros((N, N, 2, 3))  # (N, N, 2, 3)
    for i in range(0, N):
        for j in range(0, N):
            J_se2[i, j, :, :] = np.array([[1.0, 0.0, -(j - ((N - 1) / 2))], [0.0, 1.0, i - ((N - 1) / 2)]])
            grad_sum_se2 += np.matmul(grad[i, j, :], J_se2[i, j, :, :])

    J_pi = np.zeros((N, N, 3))
    for i in range(0, N):
        for j in range(0, N):
            J_pi[i, j, :] = (
                N
                * N
                * (np.matmul(grad[i, j, :], J_se2[i, j, :, :]) * intesity_sum - intesity[i, j] * grad_sum_se2)
                / (intesity_sum**2)
            )

    J_pi = J_pi.reshape(N * N, 3)

    H_se2 = np.matmul(J_pi.transpose(), J_pi) * 20.0

    # if singular return dummy cov
    try:
        sigma_se2 = np.linalg.inv(H_se2)
    except np.linalg.LinAlgError as err:
        return np.eye(2)

    return sigma_se2[:-1, :-1]
