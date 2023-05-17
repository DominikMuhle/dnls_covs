import logging
from typing import List, Tuple, overload

import torch
from PIL import Image

from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as F

logger = logging.getLogger("cropping")
logger.setLevel(logging.INFO)


class RandomCropWithIntrinsics(RandomCrop):
    # TODO: Viszualize focal point in image, to check the adapt intrinsics
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def single_forward(
        self, img: torch.Tensor | Image.Image, K_matrix: torch.Tensor, i: int, j: int, h: int, w: int
    ) -> Tuple[torch.Tensor | Image.Image, torch.Tensor]:
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            K_matrix (torch.Tensor): Calibration matrix

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        logger.debug(f"Cropping image with {i}, {j}, {h}, {w}")
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        return F.crop(img, i, j, h, w), self.adapt_intrisics(K_matrix, i, j, h, w)

    @staticmethod
    def adapt_intrisics(K_matrix: torch.Tensor, i: int, j: int, h: int, w: int):
        offset = torch.tensor([[0, 0, j], [0, 0, i], [0, 0, 0]])
        logger.debug(f"cropping image at top: {i}, left: {j}")
        logger.debug(f"K goes from")
        logger.debug(f"{K_matrix.to('cpu').numpy()}")
        logger.debug(f"to")
        logger.debug(f"{(K_matrix - offset).to('cpu').numpy()}")
        return K_matrix - offset

    @overload
    def forward(
        self, img: torch.Tensor | Image.Image, K_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor | Image.Image, torch.Tensor]:
        ...

    @overload
    def forward(
        self, img: List[torch.Tensor] | List[Image.Image], K_matrix: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor] | List[Image.Image], List[torch.Tensor]]:
        ...

    def forward(
        self,
        img: torch.Tensor | Image.Image | List[torch.Tensor] | List[Image.Image],
        K_matrix: torch.Tensor | List[torch.Tensor],
    ) -> (
        Tuple[torch.Tensor | Image.Image, torch.Tensor]
        | Tuple[List[torch.Tensor] | List[Image.Image], List[torch.Tensor]]
    ):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """

        if isinstance(img, list):
            logger.debug(f"Cropping {len(img)} at the same position.")
            i, j, h, w = self.get_params(img[0], self.size)
            assert isinstance(K_matrix, list)
            cropped_img = []
            cropped_matrix = []
            for image, matrix in zip(img, K_matrix):
                result = self.single_forward(image, matrix, i, j, h, w)
                cropped_img.append(result[0])
                cropped_matrix.append(result[1])
            return cropped_img, cropped_matrix

        logger.debug(f"Cropping only a single image.")
        i, j, h, w = self.get_params(img, self.size)

        return self.single_forward(img, K_matrix, i, j, h, w)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"
