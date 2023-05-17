from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class SyntheticConfig:
    training_frames: str = MISSING
    num_problems: int = MISSING
    num_points: int = MISSING
    num_epochs: int = MISSING
    individual_poses: bool = MISSING
    max_t: float = MISSING
    max_r: float = MISSING
    outdir: str = MISSING
    name: str = MISSING
