from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np


def load_matches(
    matches_dir: Path, host_idx: int, target_idx: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    file = matches_dir.joinpath(f"{host_idx}_{target_idx}.txt")
    if not file.is_file():
        return None
    correspondences = torch.from_numpy(np.genfromtxt(file, delimiter=","))
    return (
        correspondences[:, 0:2],
        correspondences[:, 2:4],
        correspondences[:, 4],
        torch.ones_like(correspondences[:, 0]),
    )


def write_matches(
    matches_dir: Path,
    host_idx: int,
    target_idx: int,
    matches: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
):
    file = matches_dir.joinpath(f"{host_idx}_{target_idx}.txt")
    matches_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        file,
        torch.concat([matches[0][0], matches[1][0], matches[2][0][..., None]], dim=1).to("cpu").detach().numpy(),
        delimiter=",",
    )
