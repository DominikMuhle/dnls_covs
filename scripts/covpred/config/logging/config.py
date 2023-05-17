from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class TensorboardConfig:
    enabled: bool = MISSING
    log_dir: str = MISSING
    name: str = MISSING
    eval_epochs: int = MISSING
    vis_epochs: int = MISSING
    loss_iterations: int = MISSING
    running_average_length: int = MISSING


@dataclass
class LoggingConfig:
    level: str = MISSING
    to_file: bool = MISSING
    tensorboard: TensorboardConfig = MISSING
