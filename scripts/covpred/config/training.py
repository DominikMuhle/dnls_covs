from dataclasses import dataclass

from omegaconf import MISSING

from covpred.config.pose_estimation.config import PoseEstimationConfig

from .dataset.config import DatasetConfig
from .logging.config import LoggingConfig
from .model.config import ModelConfig
from .hyperparameter.config import HyperparamterConfig
from .matching.config import MatchingConfig
from .theseus.config import TheseusConfig


@dataclass
class Config:
    logging: LoggingConfig = MISSING
    model: ModelConfig = MISSING
    dataset: DatasetConfig = MISSING
    hyperparameter: HyperparamterConfig = MISSING
    matching: MatchingConfig = MISSING
    theseus: TheseusConfig = MISSING
    pose_estimation: PoseEstimationConfig = MISSING
