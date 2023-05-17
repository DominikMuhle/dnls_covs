from dataclasses import dataclass
from typing import List

from omegaconf import MISSING

from covpred.config.pose_estimation.config import PoseEstimationConfig

from .dataset.config import DatasetConfig
from .model.config import ModelConfig
from .matching.config import MatchingConfig
from .theseus.config import TheseusConfig


@dataclass
class OptimizationCheckConfig:
    models: ModelConfig = MISSING
    dataset: DatasetConfig = MISSING
    matching: MatchingConfig = MISSING
    theseus: TheseusConfig = MISSING
    pose_estimation: PoseEstimationConfig = MISSING
    check_images: List[int] = MISSING
