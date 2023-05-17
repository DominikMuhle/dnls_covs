from typing import List, Optional, Tuple, Union
from matplotlib import pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import torch
import seaborn as sns

from covpred.visualization.common import VisualizationImage, confidence_ellipse, INCH_PER_PIXEL


def keypoints_in_img(
    imgs: VisualizationImage | List[VisualizationImage], point_number: bool = False, **kwargs
) -> Tuple[Figure, List[Axes]]:
    color = kwargs.get("color", "black")
    focal_color = kwargs.get("focal_color", "blue")
    kp_size = kwargs.get("kp_size", 1.0)
    scale = kwargs.get("scale", 3.0)
    linewidth = kwargs.get("linewidth", 1.0)
    if isinstance(imgs, VisualizationImage):
        imgs = [imgs]

    default_figsize = (
        len(imgs) * INCH_PER_PIXEL * imgs[0].img.shape[1] + (len(imgs) - 1) * 0.1,
        INCH_PER_PIXEL * imgs[0].img.shape[0],
    )
    figsize = kwargs.get("figsize", default_figsize)

    def create_img_viz(
        ax: Axes,
        img: torch.Tensor,
        K_matrix: torch.Tensor | None,
        keypoints: torch.Tensor | None,
        covariances: torch.Tensor | None,
    ):
        # works because img is flipped
        ax.imshow(img.expand(-1, -1, 3).to("cpu"))
        if K_matrix is not None:
            ax.scatter(K_matrix[0, 2], K_matrix[1, 2], s=10, c=focal_color, marker="x")
        if keypoints is not None:
            ax.scatter(
                keypoints[:, 0].to("cpu"),
                keypoints[:, 1].to("cpu"),
                c=color,
                s=kp_size,
            )
            if point_number:
                for idx, kp in enumerate(keypoints):
                    plt.text(kp[0].item(), kp[1].item(), str(idx), color=color)

            if covariances is not None:
                for keypoint, cov in zip(keypoints, covariances):
                    confidence_ellipse(
                        (keypoint[0].item(), keypoint[1].item()),
                        cov.to("cpu"),
                        ax,
                        scale=scale,
                        linewidth=linewidth,
                        edgecolor=color,
                    )

        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)

    fig, axs = plt.subplots(1, len(imgs), figsize=figsize)
    for img, ax in zip(imgs, axs):
        ax.axis("off")
        create_img_viz(ax, img.img, img.K_matrix, img.keypoints, img.covariances)
    plt.tight_layout()

    return fig, axs


def covariance_grid(
    covariances: torch.Tensor,  # (N, M, 2, 2)
    ground_truth_covariances: Optional[torch.Tensor] = None,  # (N, M, 2, 2)
    resize_covariances: bool = False,
    **kwargs
) -> Tuple[Figure, List[Axes]]:
    N, M = covariances.shape[0], covariances.shape[1]
    default_figsize = (M * INCH_PER_PIXEL * 100 + (M - 1) * 0.1, N * INCH_PER_PIXEL * 100 + (N - 1) * 0.1)
    figsize = kwargs.get("figsize", default_figsize)

    plt.style.use("seaborn")
    sns.set_context("talk")
    sns.set_style("white")
    plt.style.use("/usr/wiss/muhled/Documents/projects/deep_uncertainty_prediction/code/scripts/covpred/tex.mplstyle")
    plt.close("all")

    fig, axes = plt.subplots(N, M, figsize=figsize)

    # rehape to iterate over them
    covariances = covariances.reshape((-1, 2, 2)).to("cpu")
    if ground_truth_covariances is not None:
        ground_truth_covariances = ground_truth_covariances.reshape((-1, 2, 2)).to("cpu")

    if resize_covariances and ground_truth_covariances is not None:
        cov_scale = torch.mean(covariances[..., 0, 0] + covariances[..., 1, 1])
        gt_cov_scale = torch.mean(ground_truth_covariances[..., 0, 0] + ground_truth_covariances[..., 1, 1])
        covariances *= gt_cov_scale / cov_scale

    center = (0.0, 0.0)
    ax_lim = 1.2
    axes = np.array(axes)
    for i, ax in enumerate(axes.flat):
        ax.axis("equal")
        confidence_ellipse(
            center,
            covariances[i, ...],
            ax,
            1.0 / 3.0,
            edgecolor="red",
            linewidth=1.0,
        )

        if ground_truth_covariances is not None:
            confidence_ellipse(
                center,
                ground_truth_covariances[i, ...],
                ax,
                1.0 / 3.0,
                edgecolor="green",
                linewidth=1.0,
            )

        ax.set_xlim(-ax_lim, ax_lim)
        ax.set_ylim(-ax_lim, ax_lim)
        # ax.set_xticks([-1, 0, 1])
        # ax.set_yticks([-1, 0, 1])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    # plt.show()
    return fig, axes.tolist()
