from pathlib import Path
from typing import List, Tuple

import torch

from covpred.dataset.common import image_indices
from covpred.dataset.precomp_matches import PreCompMatchesDataset
from covpred.dataset.live_matches import LiveMatchesDataset
from covpred.config.dataset.config import DatasetConfig
from covpred.io.common import load_camera_calibration_kitti_raw, load_poses
import covpred.matching as matching


def get_img_path(dataset_directory: Path, sequence: str, image_folder: str, idx: int) -> Path:
    image_name = f"{str(idx).zfill(10)}.png"
    return dataset_directory.joinpath(sequence, image_folder, image_name)


def load_sequence(
    sequence_dir: Path, image_folder: str, matches_dir: Path
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    K_matrix = load_camera_calibration_kitti_raw(sequence_dir.parent())
    poses = load_poses(matches_dir)
    indices = image_indices(sequence_dir, image_folder)
    return poses, K_matrix, indices


class KITTI360PreCompDataset(PreCompMatchesDataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        split: str,
        matching_function: matching.MatchingFunction,
        device: str = "cpu",
        override_matches: bool = False,
        random_init: bool = True,
        ransac: bool = True,
    ) -> None:
        super().__init__(dataset_config, split, matching_function, device, override_matches, random_init, ransac)

    def get_img_path(self, sequence: str, idx: int) -> Path:
        return get_img_path(
            Path(self.dataset_config.directories.dataset), sequence, self.dataset_config.directories.images, idx
        )

    def load_sequence(self, sequence_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        sequence = f"{sequence_dir.parts[-2]}/{sequence_dir.parts[-1]}"
        return load_sequence(sequence_dir, self.dataset_config.directories.images, load_sequence(sequence_dir, self.dataset_config.directories.images, Path(self.dataset_config.directories.matches).joinpath(sequence))


class KITTIR360LiveDataset(LiveMatchesDataset):
    def __init__(self, dataset_config: DatasetConfig, split: str, device: str = "cpu") -> None:
        super().__init__(dataset_config, split, device)

    def get_img_path(self, sequence: str, idx: int) -> Path:
        return get_img_path(
            Path(self.dataset_config.directories.dataset), sequence, self.dataset_config.directories.images, idx
        )

    def load_sequence(self, sequence_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        sequence = f"{sequence_dir.parts[-2]}/{sequence_dir.parts[-1]}"
        return load_sequence(sequence_dir, self.dataset_config.directories.images, Path(self.dataset_config.directories.matches).joinpath(sequence))
