from pathlib import Path
from typing import List, Tuple

import torch

from covpred.dataset.common import image_indices
from covpred.config.pose_estimation.config import NECConfig
from covpred.dataset.precomp_matches import PreCompMatchesDataset
from covpred.dataset.live_matches import LiveMatchesDataset
from covpred.config.dataset.config import DatasetConfig
from covpred.dataset.visualization import VisDataset
from covpred.io.common import load_camera_calibration_kitti, load_poses
import covpred.matching as matching


def get_img_path(dataset_directory: Path, sequence: str, image_folder: str, idx: int) -> Path:
    image_name = f"{str(idx).zfill(6)}.png"
    return dataset_directory.joinpath(sequence, image_folder, image_name)


def get_colored_img_path(dataset_directory: Path, sequence: str, image_folder: str, idx: int) -> Path:
    image_name = f"{str(idx).zfill(6)}.jpg"
    return dataset_directory.joinpath(sequence, image_folder, image_name)


def load_sequence(
    sequence_dir: Path, image_folder: str, poses_filename: str
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    K_matrix = load_camera_calibration_kitti(sequence_dir)
    poses = load_poses(sequence_dir, poses_filename)
    indices = image_indices(sequence_dir, image_folder)
    return poses, K_matrix, indices


class KITTIPreCompDataset(PreCompMatchesDataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        nec_config: NECConfig,
        split: str,
        matching_fn: matching.MatchingFunction,
        device: str = "cpu",
        override_matches: bool = False,
        random_init: bool = True,
    ) -> None:
        super().__init__(dataset_config, nec_config, split, matching_fn, device, override_matches, random_init)

    def get_img_path(self, sequence: str, idx: int) -> Path:
        return get_img_path(
            Path(self.dataset_config.directories.dataset), sequence, self.dataset_config.directories.images, idx
        )

    def load_sequence(self, sequence_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        return load_sequence(
            sequence_dir, self.dataset_config.directories.images, self.dataset_config.directories.poses_filename
        )


class KITTILiveDataset(LiveMatchesDataset):
    def __init__(self, dataset_config: DatasetConfig, split: str, device: str = "cpu") -> None:
        super().__init__(dataset_config, split, device)

    def get_img_path(self, sequence: str, idx: int) -> Path:
        return get_img_path(
            Path(self.dataset_config.directories.dataset), sequence, self.dataset_config.directories.images, idx
        )

    def load_sequence(self, sequence_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        return load_sequence(
            sequence_dir, self.dataset_config.directories.images, self.dataset_config.directories.poses_filename
        )


class KITTIVisualization(VisDataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        colored_path: str,
        colored_image_dir: str,
        split: str,
        matching_fn: matching.MatchingFunction,
        device: str = "cpu",
    ) -> None:
        super().__init__(dataset_config, colored_path, colored_image_dir, split, matching_fn, device)

    def get_img_paths(self, sequence: str, idx: int) -> Tuple[Path, Path]:
        return get_img_path(
            Path(self.dataset_config.directories.dataset), sequence, self.dataset_config.directories.images, idx
        ), get_colored_img_path(Path(self.colored_path), sequence, self.colored_image_dir, idx)

    def load_sequence(self, sequence_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        return load_sequence(
            sequence_dir, self.dataset_config.directories.images, self.dataset_config.directories.poses_filename
        )
