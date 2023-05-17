from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class SuperPointConfig:
    max_keypoints: int = MISSING
    keypoint_threshold: float = MISSING
    nms_radius: int = MISSING


@dataclass
class SuperGlueConfig:
    weights: str = MISSING
    sinkhorn_iterations: int = MISSING
    match_threshold: float = MISSING


@dataclass
class MatchingConfig:
    superpoint: SuperPointConfig = MISSING
    superglue: SuperGlueConfig = MISSING
