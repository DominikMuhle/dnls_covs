from typing import Optional

from matplotlib import pyplot as plt
import seaborn as sns
import torch
import theseus as th

from math.pose_optimization import evaluate_around_minimum, RPY


def mini_plot(
    host_bvs: torch.Tensor,
    host_bvs_covs: Optional[torch.Tensor],
    target_bvs: torch.Tensor,
    target_bvs_covs: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
    gt_pose: th.SE3,
    regularization: float,
    device: str,
):
    axis = (RPY.pitch, RPY.yaw)
    x_ticks = torch.linspace(-0.1, 0.1, 101).to(device).type_as(host_bvs)
    y_ticks = torch.linspace(-0.05, 0.05, 101).to(device).type_as(host_bvs)
    grid_x, grid_y, costs = evaluate_around_minimum(
        host_bvs,
        host_bvs_covs,
        target_bvs,
        target_bvs_covs,
        weights,
        th.SE3(tensor=gt_pose.tensor[0][None]),
        regularization,
        axis,
        x_ticks,
        y_ticks,
        degree=True,
    )
    # make to degree
    grid_x = grid_x * (180.0 / torch.pi)
    grid_y = grid_y * (180.0 / torch.pi)

    arg_min = (costs == torch.min(costs)).nonzero()
    min_x, min_y = (
        grid_x[arg_min[0, 0].item(), arg_min[0, 1].item()].to("cpu"),
        grid_y[arg_min[0, 0].item(), arg_min[0, 1].item()].to("cpu"),
    )

    plt.style.use("seaborn")
    sns.set_context("talk")
    sns.set_style("white")
    plt.style.use("/usr/wiss/muhled/Documents/projects/deep_uncertainty_prediction/code/scripts/de_un_pre/tex.mplstyle")
    cmap = sns.color_palette("viridis", as_cmap=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.pcolormesh(grid_x.to("cpu"), grid_y.to("cpu"), costs.to("cpu"), cmap=cmap)
    im.set_edgecolor("face")
    if (host_bvs_covs is not None) or (target_bvs_covs is not None):
        ax.set_title(r"$E_{PNEC}$")
    elif weights is not None:
        ax.set_title(r"$E_{wNEC}$")
    else:
        ax.set_title(r"$E_{NEC}$")

    ax.set_xlabel("pitch in [deg]")
    ax.set_ylabel("yaw in [deg]")
    ax.set_aspect("equal")
    ax.scatter(
        grid_x[arg_min[0, 0].item(), arg_min[0, 1].item()].to("cpu"),
        grid_y[arg_min[0, 0].item(), arg_min[0, 1].item()].to("cpu"),
        c="r",
        s=100.0,
        marker=".",
    )
    ax.scatter(
        0,
        0,
        c="g",
        s=100.0,
        marker=".",
    )
    ax.set_xticks([-0.1, 0, 0.1])
    ax.set_yticks([-0.05, 0, 0.05])
