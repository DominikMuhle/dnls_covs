from typing import Tuple
from dataclasses import dataclass

from torch.utils.data import Dataset
import torch
import theseus as th


@dataclass
class SyntheticFrame:
    pts: torch.Tensor  # M,N,2
    noisy_pts: torch.Tensor  # M,N,2
    covs: torch.Tensor  # M,N,2,2
    init_covs: torch.Tensor  # M,N,2,2


@dataclass
class SyntheticFramePairs:
    host: SyntheticFrame
    target: SyntheticFrame


class SyntheticDataset(Dataset):
    def __init__(
        self,
        image_pairs: SyntheticFramePairs,
        gt_poses: th.SE3,
    ) -> None:
        self.image_pairs = image_pairs
        self.gt_poses = gt_poses  # M,3,3

    def __len__(self) -> int:
        return self.gt_poses.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # print(index)
        if torch.is_tensor(index):
            index = index.tolist()

        return (
            self.image_pairs.host.noisy_pts[index],
            self.image_pairs.target.noisy_pts[index],
            self.gt_poses.tensor[index],
        )
