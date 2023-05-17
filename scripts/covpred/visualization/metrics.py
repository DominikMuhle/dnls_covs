from typing import Dict, List

from matplotlib import pyplot as plt
import torch
import seaborn as sns


def PartialRPE1Plot(
    partial_rpe1s: Dict[str, List[float]],
    colors: List,
    linestyles: List,
    figsize,
):
    if len(partial_rpe1s) != len(colors):
        print("Didn't provide the correct number of colors. Will revert to use the seaborn bright color palette")
        colors = sns.color_palette("bright")[: len(partial_rpe1s)]

    assert len(colors) == len(linestyles)
    plt.style.use("seaborn")
    sns.set_context("talk")
    sns.set_style("white")

    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for (name, partial_rpe1), color, linestyle in zip(partial_rpe1s.items(), colors, linestyles):
        ax.plot(
            partial_rpe1,
            c=color,
            label=name,
            linestyle=linestyle,
        )

    ax.set_ylim(0.0, 0.5)
    # ax.axis("equal")
    ax.legend(prop={"size": 6}, handlelength=2.0)

    return fig, ax


def CDF(
    errors: Dict[str, torch.Tensor],
    colors: List,
    linestyles: List,
    figsize,
):
    if len(errors) != len(colors):
        print("Didn't provide the correct number of colors. Will revert to use the seaborn bright color palette")
        colors = sns.color_palette("bright")[: len(errors)]

    assert len(colors) == len(linestyles)
    plt.style.use("seaborn")
    sns.set_context("talk")
    sns.set_style("white")

    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for (name, e_r), color, linestyle in zip(errors.items(), colors, linestyles):
        sorted_errors = [0.0]
        sorted_errors.extend(e_r.sort()[0].tolist())
        percentage = torch.linspace(0.0, 1.0, len(sorted_errors)).tolist()
        ax.plot(
            sorted_errors,
            percentage,
            c=color,
            label=name,
            linestyle=linestyle,
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 0.5)
    # ax.axis("equal")
    ax.legend(prop={"size": 6}, handlelength=2.0)

    return fig, ax
