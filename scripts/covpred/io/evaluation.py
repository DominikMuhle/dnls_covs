import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from covpred.io.common import load_poses, save_poses


def load_pre_computed_poses(
    path: Path,
    filename: str,
) -> Optional[List[torch.Tensor]]:
    if not path.joinpath(filename).is_file():
        return None
    print(f"Loading precomputed proses from {str(path)}")
    return [pose for pose in load_poses(path, filename)]


def load_metrics(
    path: Path,
) -> Optional[Dict[str, float]]:
    if not path.is_file():
        return None
    print(f"Loading metrics from {str(path)}")
    with open(path, "r") as f:
        metrics = json.load(f)
    return metrics


def save_live_methods(path: Path, poses: List[torch.Tensor]):
    path.parent.mkdir(parents=True, exist_ok=True)
    # path.mkdir(parents=True, exist_ok=True)
    save_poses(path, torch.stack(poses, dim=0))


def save_metrics(
    path: Path,
    metrics: Dict[str, float],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving metrics at {str(path)}")
    with open(path, "w") as fp:
        json.dump(metrics, fp, indent="\t")
