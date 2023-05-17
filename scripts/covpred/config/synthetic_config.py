from dataclasses import dataclass

from omegaconf import MISSING
from covpred.config.pose_estimation.config import PoseEstimationConfig

from covpred.config.synthetic.config import SyntheticConfig

from .logging.config import LoggingConfig
from .theseus.config import TheseusConfig


@dataclass
class SyntheticExperimentConfig:
    logging: LoggingConfig = MISSING
    theseus: TheseusConfig = MISSING
    pose_estimation: PoseEstimationConfig = MISSING
    synthetic: SyntheticConfig = MISSING
