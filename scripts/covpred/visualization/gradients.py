from typing import List, Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import torch


def gradient_vis(gradients: List[torch.Tensor], **kwargs) -> Tuple[Figure, Axes]:
    default_figsize = (
        0.1,
        0.1,
    )
    figsize = kwargs.get("figsize", default_figsize)

    L, V = torch.linalg.eig(torch.stack(gradients)[:, 0, :])
    V_real = V.real

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    scale = 2.0
    ax.scatter(V_real[..., 0, 1], V_real[..., 1, 1], c=range(V_real.shape[0], 0, -1), cmap="autumn", s=scale)
    ax.scatter(V_real[..., 0, 0], V_real[..., 1, 0], c=range(V_real.shape[0], 0, -1), cmap="winter", s=scale)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("equal")

    return fig, ax
