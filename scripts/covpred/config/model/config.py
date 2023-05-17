from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import MISSING


@dataclass
class FilterConfig:
    name: str = MISSING
    args = {}


@dataclass
class ModelOutputConfig:
    filter1: FilterConfig = MISSING
    filter2: FilterConfig = MISSING
    filter3: FilterConfig = MISSING

    representation: str = MISSING


@dataclass
class ModelConfig:
    path: str = MISSING

    base_model: str = MISSING
    date: Optional[str] = MISSING
    checkpoint: Optional[str] = MISSING

    override_filters: bool = MISSING
    resume_training: bool = MISSING
    output: ModelOutputConfig = MISSING
    klt_scaling: float = MISSING
    isotropic_covariances: bool = MISSING
