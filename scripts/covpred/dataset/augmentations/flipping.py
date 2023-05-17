from typing import Tuple

import torch
import theseus as th
from PIL import Image


class ImageFlip(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, imgs: torch.Tensor, K_matrix: torch.Tensor, rel_poses: th.SE3
    ) -> Tuple[torch.Tensor, torch.Tensor, th.SE3]:
        """Flips the image horizontally and adapts the camera-calibration matrix and adapts the relative pose between the images

        Args:
            imgs (torch.Tensor): N,C,H,W image tensor
            K_matric (torch.Tensor): N,3,3 cameras calibration matrices
            rel_poses (th.SE3): P,3,4 relative poses

        Returns:
            Tuple[torch.Tensor, torch.Tensor, th.SE3]: _description_
        """
        H, W = imgs.shape[-2], imgs.shape[-1]

        imgs = torch.flip(imgs, (-1,))

        # Warning: Does not account for s
        K_matrix[:, 0, 2] = W - K_matrix[:, 0, 2]

        poses = rel_poses.tensor
        poses[:, 0, :] = -poses[:, 0, :]
        poses[:, :, 0] = -poses[:, :, 0]

        return imgs, K_matrix, th.SE3(tensor=poses)
