from abc import abstractmethod
from math import sqrt
from pathlib import Path
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import theseus as th
from torchvision.transforms import ColorJitter

from covpred.common import load_img
from covpred.config.dataset.config import DatasetConfig
from covpred.dataset.augmentations.flipping import ImageFlip
from covpred.dataset.augmentations.random_crop import RandomCropWithIntrinsics
from covpred.dataset.common import ImgData, ImgTuple, create_idx_tuples


LiveData = Tuple[
    torch.Tensor,  # images
    torch.Tensor,  # K_matrix
    torch.Tensor,  # rel_poses
    torch.Tensor,  # host_img_idx
    torch.Tensor,  # target_img_idx
]

import matplotlib.pyplot as plt


def plot_img_with_focal_point(img: torch.Tensor, K_matrix: torch.Tensor, suffix: str = ""):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img.to("cpu"))

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)

    ax.scatter(K_matrix[0, 2], K_matrix[1, 2], s=10, c="r")

    plt.savefig(f"image_{suffix}.png")


class LiveMatchesDataset(Dataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        split: str,
        device: str = "cpu",
    ) -> None:
        if split not in dataset_config.directories.sequences.keys():
            raise ValueError(
                f"Split not available in the given sequences orderings. Available splits: {dataset_config.directories.sequences.keys()}. Name provided: {split}"
            )

        self.dataset_config = dataset_config
        self.split = split
        self.device = device
        sequences = self.dataset_config.directories.sequences[split]
        if isinstance(sequences, str):
            sequences = [sequences]
        self.sequences = sequences
        self.img_precision = torch.float32
        self.random_crop = RandomCropWithIntrinsics(self.dataset_config.image_augmentations.crop_size)
        brightness = (
            self.dataset_config.image_augmentations.brightness[0],
            self.dataset_config.image_augmentations.brightness[1],
        )
        contrast = (
            self.dataset_config.image_augmentations.contrast[0],
            self.dataset_config.image_augmentations.contrast[1],
        )
        saturation = (
            self.dataset_config.image_augmentations.saturation[0],
            self.dataset_config.image_augmentations.saturation[1],
        )
        hue = (self.dataset_config.image_augmentations.hue[0], self.dataset_config.image_augmentations.hue[1])
        self.color_jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.flip = ImageFlip()

        self.img_tuples = self.create_img_tuple()

    def __len__(self) -> int:
        return len(self.img_tuples)

    @abstractmethod
    def get_img_path(self, sequence: str, idx: int) -> Path:
        pass

    @abstractmethod
    def load_sequence(self, sequence_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        pass

    def create_img_tuple(self) -> List[ImgTuple]:
        img_tuples = []
        sequences = self.dataset_config.directories.sequences[self.split]
        for sequence in sequences:
            seq_dir = Path(self.dataset_config.directories.dataset).joinpath(sequence)

            poses, K_matrix, indices = self.load_sequence(seq_dir)

            idx_tuples = create_idx_tuples(
                indices,
                self.dataset_config.sweep.tuple_length,
                self.dataset_config.sweep.step_size,
                self.dataset_config.sweep.skip,
                self.dataset_config.sweep.max_idx,
                self.dataset_config.sweep.tuple_per_base_img,
            )

            for idx_tuple in idx_tuples:
                images: List[ImgData] = []
                pairs: List[Tuple[int, int]] = []
                for img_idx in idx_tuple:
                    images.append(ImgData(sequence, img_idx, K_matrix, th.SE3(tensor=poses[img_idx][None])))
                for tuple_idx_host, _ in enumerate(idx_tuple):
                    if tuple_idx_host + 1 == 2 and len(idx_tuple) == 2:
                        break

                    if tuple_idx_host + 1 == len(idx_tuple):
                        tuple_idx_target = 0
                    else:
                        tuple_idx_target = tuple_idx_host + 1

                    pairs.append((tuple_idx_host, tuple_idx_target))
                img_tuples.append(ImgTuple(images, pairs))
        return img_tuples

    def __getitem__(self, index) -> LiveData:
        if torch.is_tensor(index):
            index = index.tolist()

        sequence = self.img_tuples[index].images[0].sequence
        K_matrices = []
        images = []
        poses = []
        for img in self.img_tuples[index].images:
            image = load_img(self.get_img_path(sequence, img.idx), self.img_precision)

            # if self.dataset_config.image_augmentations.color_jitter:
            #     image = self.color_jitter(image)

            if self.dataset_config.image_augmentations.noise:
                noise_intensity = torch.rand(1, 1) * sqrt(self.dataset_config.image_augmentations.noise_var)
                image = torch.clip(image + noise_intensity * torch.randn_like(image), min=0.0, max=1.0)

            images.append(image)
            poses.append(img.pose.tensor)
            K_matrices.append(img.K_matrix)

        host_idx = []
        target_idx = []
        rel_poses = []
        for tuple_idx_host, tuple_idx_target in self.img_tuples[index].pairs:
            host_idx.append(tuple_idx_host)
            target_idx.append(tuple_idx_target)
            rel_poses.append(
                th.SE3(tensor=poses[tuple_idx_host]).between(th.SE3(tensor=poses[tuple_idx_target])).tensor
            )

        # cropping
        cropped_images = []
        cropped_matrices = []
        if self.dataset_config.image_augmentations.random_crop:
            individual_crops = False
            if individual_crops:
                for image, K_matrix in zip(images, K_matrices):
                    result = self.random_crop(image, K_matrix)
                    cropped_images.append(result[0])
                    cropped_matrices.append(result[1])
            else:
                cropped_images, cropped_matrices = self.random_crop(images, K_matrices)
        else:
            for image, K_matrix in zip(images, K_matrices):
                cropped_images.append(
                    image[
                        ...,
                        : self.dataset_config.image_augmentations.crop_size[0],
                        : self.dataset_config.image_augmentations.crop_size[1],
                    ]
                )
                cropped_matrices.append(K_matrix)

        cropped_images = torch.stack(cropped_images, dim=0)
        cropped_matrices = torch.stack(cropped_matrices, dim=0)
        rel_poses = th.SE3(tensor=torch.concat(rel_poses, dim=0))

        if self.dataset_config.image_augmentations.flipping:
            flip = bool(random.getrandbits(1))
            # if flip:
            #     cropped_images, cropped_matrices, rel_poses = self.flip(
            #         cropped_images,
            #         cropped_matrices,
            #         rel_poses,
            #     )
            #     pass

        return (
            cropped_images,
            torch.linalg.inv(cropped_matrices),
            rel_poses.tensor,
            torch.tensor(host_idx, dtype=torch.int),
            torch.tensor(target_idx, dtype=torch.int),
        )
