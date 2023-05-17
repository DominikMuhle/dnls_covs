from dataclasses import dataclass
from typing import Dict, Any

import theseus as th
from omegaconf import MISSING


@dataclass
class OptimizerConfig:
    name: str = MISSING
    # TODO: integrate both, research options for theseus
    # linearization: str = MISSING
    # linear_solver: str = MISSING
    kwargs: Dict[str, Any] = MISSING


@dataclass
class TheseusConfig:
    optimizer: OptimizerConfig = MISSING

    step_size: float = MISSING
    EPS: float = MISSING
    abs_err_tolerance: float = MISSING
    rel_err_tolerance: float = MISSING
