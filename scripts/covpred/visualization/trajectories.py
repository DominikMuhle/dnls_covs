from math import cos, sin
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from covpred.math.common import rotation_matrix_2d


def Trajectory2D(
    ground_truth: torch.Tensor,
    axes: Tuple,
    estimated: Dict[str, torch.Tensor],
    colors: List,
    linestyles: List[str],
    figsize: Tuple[float, float],
    trajectory_args: Dict = {},
    axis_args: Dict = {},
) -> Tuple[Figure, Axes]:
    if len(estimated) != len(colors):
        print("Didn't provide the correct number of colors. Will revert to use the seaborn bright color palette")
        colors = sns.color_palette("bright")[: len(estimated)]

    assert len(colors) == len(linestyles)
    plt.style.use("seaborn")
    sns.set_context("talk")
    sns.set_style("white")
    # plt.style.use("scripts/tex.mplstyle")

    helper = {"x": 0, "y": 1, "z": 2}
    ax1 = helper[axes[0][-1]]
    ax1_sign = -1 if axes[0][0] == "-" else 1
    ax2 = helper[axes[1][-1]]
    ax2_sign = -1 if axes[1][0] == "-" else 1

    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gt_translation = ground_truth[..., :3, 3]
    color_gt = [0, 0, 0]
    ax.plot(
        ax1_sign * gt_translation[:, ax1],
        ax2_sign * gt_translation[:, ax2],
        c=color_gt,
        label="ground truth",
        **trajectory_args
    )

    for (name, result), color, linestyle in zip(estimated.items(), colors, linestyles):
        translation = result[..., :3, 3]
        ax.plot(
            ax1_sign * translation[:, ax1],
            ax2_sign * translation[:, ax2],
            c=color,
            label=name,
            linestyle=linestyle,
            **trajectory_args
        )

    ax.axis("equal")
    ax.legend(prop={"size": 12}, handlelength=2.5, loc="upper left")
    ax.set(**axis_args)

    return fig, ax


def IndividualTrajectories(
    ground_truth: torch.Tensor,
    axes: Tuple,
    rotation_angle: float,
    estimated: Dict[str, torch.Tensor],
    colors: List,
    linestyles: List,
    figsize,
    trajectory_args: Dict = {},
    axis_args: Dict = {},
) -> List[Tuple[Figure, Axes]]:
    plt.style.use("seaborn")
    sns.set_context("talk")
    sns.set_style("white")
    plt.style.use("/usr/wiss/muhled/Documents/projects/deep_uncertainty_prediction/code/scripts/de_un_pre/tex.mplstyle")

    if len(estimated) != len(colors):
        print("Didn't provide the correct number of colors. Will revert to use the seaborn bright color palette")
        colors = sns.color_palette("bright")[: len(estimated)]

    assert len(colors) == len(linestyles)

    helper = {"x": 0, "y": 1, "z": 2}
    ax1 = helper[axes[0][-1]]
    ax1_sign = -1 if axes[0][0] == "-" else 1
    ax2 = helper[axes[1][-1]]
    ax2_sign = -1 if axes[1][0] == "-" else 1

    rotation = torch.tensor(
        [[cos(rotation_angle), -sin(rotation_angle)], [sin(rotation_angle), cos(rotation_angle)]],
        dtype=ground_truth.dtype,
        device=ground_truth.device,
    )

    figures: List[Tuple[Figure, Axes]] = []
    for (name, result), color, linestyle in zip(estimated.items(), colors, linestyles):
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        gt_translation = ground_truth[..., :3, 3]
        xy = torch.stack(
            [
                ax1_sign * gt_translation[:, ax1],
                ax2_sign * gt_translation[:, ax2],
            ]
        )
        rot_xy = torch.matmul(rotation, xy)

        color_gt = [0, 0, 0]
        ax.plot(rot_xy[0], rot_xy[1], c=color_gt, label="ground truth", **trajectory_args)

        translation = result[..., :3, 3]
        est_xy = torch.stack(
            [
                ax1_sign * translation[:, ax1],
                ax2_sign * translation[:, ax2],
            ]
        )
        est_rot_xy = torch.matmul(rotation, est_xy)
        ax.plot(est_rot_xy[0], est_rot_xy[1], c=color, label=name, linestyle=linestyle, **trajectory_args)

        ax.axis("equal")
        # ax.legend(prop={"size": 6}, handlelength=1.5)
        ax.set(**axis_args)

        figures.append((fig, ax))
        # plt.tight_layout()
        # plt.savefig(base_path.joinpath(name + "trajectory_teaser.pdf"), bbox_inches="tight")

    return figures


def VideoTrajectories(
    ground_truth: torch.Tensor,
    axes: Tuple,
    estimated: Dict[str, torch.Tensor],
    idx: int,
    max_idx: int,
    colors: List,
    linestyles: List[str],
    figsize: Tuple[float, float],
    trajectory_args: Dict = {},
    axis_args: Dict = {},
) -> Tuple[Figure, Axes]:
    if len(estimated) != len(colors):
        print("Didn't provide the correct number of colors. Will revert to use the seaborn bright color palette")
        colors = sns.color_palette("bright")[: len(estimated)]

    assert len(colors) == len(linestyles)
    plt.style.use("seaborn")
    sns.set_context("talk")
    sns.set_style("white")
    # plt.style.use("scripts/tex.mplstyle")

    helper = {"x": 0, "y": 1, "z": 2}
    ax1 = helper[axes[0][-1]]
    ax1_sign = -1 if axes[0][0] == "-" else 1
    ax2 = helper[axes[1][-1]]
    ax2_sign = -1 if axes[1][0] == "-" else 1

    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gt_translation = ground_truth[..., :3, 3]

    gt_2d = torch.stack([ax1_sign * gt_translation[:, ax1], ax2_sign * gt_translation[:, ax2]], dim=-1)

    last_t = gt_2d[idx] - gt_2d[idx - 1]

    offset = gt_2d[idx]
    angle = torch.arctan2(last_t[0], last_t[1]).item()

    time_frame = 50
    if max_idx - idx < time_frame:
        angle = (max_idx - idx) / time_frame * angle
    orientation = rotation_matrix_2d(angle).type_as(gt_2d)
    trunc_translation = torch.einsum("ij,kj->ki", orientation, gt_2d[:idx] - offset)

    color_gt = [0, 0, 0]
    ax.plot(trunc_translation[:, 0], trunc_translation[:, 1], c=color_gt, label="ground truth", **trajectory_args)

    translations_2d = {}
    for (name, result), color, linestyle in zip(estimated.items(), colors, linestyles):
        translation = result[..., :3, 3]

        translation_2d = torch.stack([ax1_sign * translation[:, ax1], ax2_sign * translation[:, ax2]], dim=-1)
        translations_2d[name] = translation_2d

        trunc_translation = torch.einsum("ij,kj->ki", orientation, translation_2d[:idx] - offset)
        ax.plot(
            trunc_translation[:, 0],
            trunc_translation[:, 1],
            c=color,
            label=name,
            linestyle=linestyle,
            **trajectory_args
        )

    ax.axis("equal")
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    distances = {
        name: torch.abs(torch.einsum("ij,j->i", orientation, t_2d[idx] - gt_2d[idx]))
        for name, t_2d in translations_2d.items()
    }
    max_distances = max([max(distance[0].item(), distance[1].item()) for _, distance in distances.items()])
    limit = (min(-max_distances - 2.0, -10), max(max_distances + 2.0, 10))
    curr_x_lim = ax.get_xlim()
    curr_y_lim = ax.get_ylim()

    x_lim = limit
    y_lim = limit
    if max_idx - idx < time_frame:
        lim_lambda = (max_idx - idx) / time_frame
        if lim_lambda < 0:
            lim_lambda = 0
        x_lim = (
            lim_lambda * limit[0] + (1 - lim_lambda) * (curr_x_lim[0]),
            lim_lambda * limit[1] + (1 - lim_lambda) * (curr_x_lim[1]),
        )
        y_lim = (
            lim_lambda * limit[0] + (1 - lim_lambda) * (curr_y_lim[0]),
            lim_lambda * limit[1] + (1 - lim_lambda) * (curr_y_lim[1]),
        )

    ax.set_xlim(x_lim[0], y_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])

    ax.legend(prop={"size": 12}, handlelength=2.5, loc="upper left")
    ax.set(**axis_args)

    return fig, ax
