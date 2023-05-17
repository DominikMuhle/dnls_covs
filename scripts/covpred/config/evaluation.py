from dataclasses import dataclass
from typing import List

from omegaconf import MISSING

from covpred.config.pose_estimation.config import PoseEstimationConfig

from .dataset.config import DatasetConfig
from .model.config import ModelConfig
from .hyperparameter.config import HyperparamterConfig
from .matching.config import MatchingConfig
from .theseus.config import TheseusConfig


@dataclass
class Config:
    model: ModelConfig = MISSING
    dataset: DatasetConfig = MISSING
    hyperparameter: HyperparamterConfig = MISSING
    matching: MatchingConfig = MISSING
    theseus: TheseusConfig = MISSING
    pose_estimation: PoseEstimationConfig = MISSING
    output_dir: str = MISSING
    methods: List[str] = MISSING
