from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class HyperparamterConfig:
    batch_size: int = MISSING
    learning_rate: float = MISSING
    epochs: int = MISSING
    error_threshold: float = MISSING

    accumulation_steps: int = MISSING
    use_amp: bool = MISSING

    loss_type: str = MISSING