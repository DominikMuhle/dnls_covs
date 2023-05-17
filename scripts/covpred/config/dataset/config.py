from dataclasses import dataclass
from typing import Dict, List

from omegaconf import MISSING


@dataclass
class DatasetDirectoriesConfig:
    dataset: str = MISSING
    sequences: Dict[str, List[str]] = MISSING
    poses_filename: str = MISSING
    images: str = MISSING
    matches: str = MISSING


@dataclass
class DatasetMatchingConfig:
    algorithm: str = MISSING
    max_points: int = MISSING


@dataclass
class DatasetSweepConfig:
    self_supervised: bool = MISSING
    tuple_length: int = MISSING
    step_size: int = MISSING
    skip: int = MISSING
    tuple_per_base_img: int = MISSING
    max_idx: int = MISSING


@dataclass
class ImageAugmentationConfig:
    crop_size: List[int] = MISSING
    random_crop: bool = MISSING
    color_jitter: bool = MISSING
    brightness: List[float] = MISSING
    contrast: List[float] = MISSING
    saturation: List[float] = MISSING
    hue: List[float] = MISSING
    noise: bool = MISSING
    noise_var: float = MISSING
    flipping: bool = MISSING


@dataclass
class KeypointAugmentationConfig:
    keypoint_jitter: bool = MISSING
    jitter_std: float = MISSING


@dataclass
class DatasetConfig:
    name: str = MISSING
    directories: DatasetDirectoriesConfig = MISSING

    matching: DatasetMatchingConfig = MISSING

    sweep: DatasetSweepConfig = MISSING

    image_augmentations: ImageAugmentationConfig = MISSING
    keypoint_augmentations: KeypointAugmentationConfig = MISSING
