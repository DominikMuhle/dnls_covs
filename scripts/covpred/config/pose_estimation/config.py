from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class NECConfig:
    ransac: bool = MISSING
    inlier_threshold: float = MISSING
    opt_t: bool = MISSING


@dataclass
class PoseEstimationConfig:
    regularization: float = MISSING
    scaling: float = MISSING
    nec: NECConfig = MISSING
