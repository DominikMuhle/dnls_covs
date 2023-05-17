from typing import Callable, Optional, Tuple

import torch

CovarianceFilter = Callable[[torch.Tensor], torch.Tensor]


def median_filter(n_medians: float = 2.0) -> CovarianceFilter:
    """Scale the image covariances with their median value to create a more consistent visualization in tensorboard.

    Args:
        parameter_img (torch.Tensor): B,3,H,W covariances as scale, alpha, beta of the images
        n_medians (float, optional): Set the boundary of where to clip the covariance scales to avoid images that are to bright or to dark. Defaults to 2.0.

    Returns:
        torch.Tensor: scaled image for a more consistent visualization brightness
    """

    def _filter(covariances: torch.Tensor) -> torch.Tensor:
        scales = covariances[..., 0, 0] + covariances[..., 1, 1]  # B, H, W
        medians = torch.median((scales).flatten(-2, -1), dim=-1,)[
            0
        ]  # B
        return covariances * (torch.clip(scales, None, n_medians * medians[..., None, None]) / scales)[..., None, None]

    return _filter


def mean_filter(n_means: float = 2.0) -> CovarianceFilter:
    """Scale the image covariances with their mean value to create a more consistent visualization in tensorboard.

    Args:
        parameter_img (torch.Tensor): B,3,H,W covariances as scale, alpha, beta of the images
        n_medians (float, optional): Set the boundary of where to clip the covariance scales to avoid images that are to bright or to dark. Defaults to 2.0.

    Returns:
        torch.Tensor: scaled image for a more consistent visualization brightness
    """

    def _filter(covariances: torch.Tensor) -> torch.Tensor:
        scales = covariances[..., 0, 0] + covariances[..., 1, 1]  # B, H, W
        medians = torch.mean((scales).flatten(-2, -1), dim=-1,)[
            0
        ]  # B
        return covariances * (torch.clip(scales, None, n_means * medians[..., None, None]) / scales)[..., None, None]

    return _filter


def clip_filter(upper_bound: float = 2.0) -> CovarianceFilter:
    """Scale the image covariances with their mean value to create a more consistent visualization in tensorboard.

    Args:
        parameter_img (torch.Tensor): B,3,H,W covariances as scale, alpha, beta of the images
        n_medians (float, optional): Set the boundary of where to clip the covariance scales to avoid images that are to bright or to dark. Defaults to 2.0.

    Returns:
        torch.Tensor: scaled image for a more consistent visualization brightness
    """

    def _filter(covariances: torch.Tensor) -> torch.Tensor:
        scales = covariances[..., 0, 0] + covariances[..., 1, 1]  # B, H, W
        return covariances * (torch.clip(scales, None, upper_bound) / scales)[..., None, None]

    return _filter


def scale_parameter_img(parameter_img: torch.Tensor, n_medians: float = 2.0) -> torch.Tensor:
    """Scale the image covariances with their median value to create a more consistent visualization in tensorboard. Scale of the covariances will be between 0 and 1

    Args:
        parameter_img (torch.Tensor): B,3,H,W covariances as scale, alpha, beta of the images
        n_medians (float, optional): Set the boundary of where to clip the covariance scales to avoid images that are to bright or to dark. Defaults to 2.0.

    Returns:
        torch.Tensor: scaled image for a more consistent visualization brightness
    """
    medians = torch.median(parameter_img[..., :, :, 0].flatten(-2, -1), dim=-1)[0]
    parameter_img[..., :, :, 0] = torch.clip(
        parameter_img[..., :, :, 0], None, n_medians * medians[..., None, None]
    ) / (n_medians * medians[..., None, None])
    return parameter_img


def scale_covariance_img(covariance_img: torch.Tensor, n_medians: float = 2.0) -> torch.Tensor:
    """Scale the image covariances with their median value to create a more consistent visualization in tensorboard.

    Args:
        covariance_img (torch.Tensor): B,H,W,2,2 covariances
        n_medians (float, optional): Set the boundary of where to clip the covariance scales to avoid images that are to bright or to dark. Defaults to 2.0.

    Returns:
        torch.Tensor: scaled image for a more consistent visualization brightness
    """
    scales = covariance_img[..., :, :, 0, 0] + covariance_img[..., :, :, 1, 1]  # B, H, W
    medians = torch.median((scales).flatten(-2, -1), dim=-1,)[
        0
    ]  # B
    return (
        covariance_img
        * (
            torch.clip(scales, None, n_medians * medians[..., None, None])
            # / (scales * n_medians * medians[..., None, None])
        )[..., None, None]
    )
