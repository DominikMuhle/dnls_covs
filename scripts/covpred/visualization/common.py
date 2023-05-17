from dataclasses import dataclass
from typing import Tuple
from matplotlib import transforms
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

import torch

INCH_PER_PIXEL = 0.01

img_frame = torch.ones(1, 350, 1200)
img_frame[:, :2, :] = 0
img_frame[:, -2:, :] = 0
img_frame[:, :, :2] = 0
img_frame[:, :, -2:] = 0


@dataclass
class VisualizationImage:
    img: torch.Tensor = img_frame
    K_matrix: torch.Tensor | None = None
    keypoints: torch.Tensor | None = None
    covariances: torch.Tensor | None = None


def confidence_ellipse(
    mean: Tuple[float, float],
    cov: torch.Tensor,
    ax: Axes,
    scale: float = 3.0,
    linewidth: float = 1.0,
    facecolor: str = "none",
    **kwargs
):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    scale : float
        Scale the covariance sizes of all for better visualization.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    n_std = 3.0
    pearson = cov[0, 1] / torch.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = torch.sqrt(1 + pearson).item()
    ell_radius_y = torch.sqrt(1 - pearson).item()
    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, linewidth=linewidth, **kwargs
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = torch.sqrt(cov[0, 0]).item() * n_std * scale
    mean_x = mean[0]

    # calculating the standard deviation of y ...
    scale_y = torch.sqrt(cov[1, 1]).item() * n_std * scale
    mean_y = mean[1]

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
