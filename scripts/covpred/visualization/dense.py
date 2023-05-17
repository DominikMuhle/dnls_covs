from typing import List, Optional, Tuple, Union
from matplotlib import pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import torch

from covpred.visualization.common import VisualizationImage, INCH_PER_PIXEL


def dense_covariance_images(imgs: VisualizationImage | List[VisualizationImage], **kwargs) -> Tuple[Figure, List[Axes]]:
    if isinstance(imgs, VisualizationImage):
        imgs = [imgs]

    default_figsize = (
        len(imgs) * INCH_PER_PIXEL * imgs[0].img.shape[1] + (len(imgs) - 1) * 10 * INCH_PER_PIXEL,
        INCH_PER_PIXEL * imgs[0].img.shape[0],
    )
    figsize = kwargs.get("figsize", default_figsize)

    def create_img_viz(ax: Axes, img: torch.Tensor):
        # works because img is flipped
        ax.imshow(img.to("cpu"))

        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)

    fig, axs = plt.subplots(1, len(imgs), figsize=figsize)
    for img, ax in zip(imgs, axs):
        ax.axis("off")
        create_img_viz(ax, img.img)

    plt.tight_layout()
    return fig, axs
