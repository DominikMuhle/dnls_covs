from enum import Enum

import torch.nn as nn

from .superpoint_architecture import SuperCov
from .unet.unet_model import UNet, UNetM, UNetSmall, UNetXS


class Models(Enum):
    SuperCov = 1
    UNet = 2
    UNetM = 3
    UNetSmall = 4
    UNetXS = 5


def get_model(model_name: Models) -> nn.Module:
    match model_name:
        case Models.SuperCov:
            return SuperCov()     
        case Models.SuperCov:
            return SuperCov()
        case Models.UNet:
            return UNet(1, 3)
        case Models.UNetM:
            return UNetM(1, 3)
        case Models.UNetSmall:
            return UNetSmall(1, 3)
        case Models.UNetXS:
            return UNetXS(1, 3)
