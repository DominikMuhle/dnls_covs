from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
import theseus as th


@dataclass
class ImgData:
    sequence: str
    idx: int
    K_matrix: torch.Tensor
    pose: th.SE3


@dataclass
class ImgPair:
    host_keypoints: torch.Tensor
    target_keypoints: torch.Tensor
    mask: torch.Tensor
    confidence: torch.Tensor
    init_pose: Optional[th.SE3] = None
    nec_pose: Optional[th.SE3] = None


@dataclass
class ImgTuple:
    images: List[ImgData]
    pairs: List[Tuple[int, int]]


@dataclass
class ImgTupleWithMatches:
    images: List[ImgData]
    pairs: Dict[Tuple[int, int], ImgPair]


def image_indices(sequence_dir: Path, images_dir: str) -> List[int]:
    indices = []
    for file in sorted(sequence_dir.joinpath(images_dir).iterdir()):
        if not file.is_file():
            continue
        stem = file.stem
        try:
            idx = int(stem)
        except:
            continue
        indices.append(idx)
    return indices


def create_idx_tuples(
    img_idx: List[int],
    tuple_length: int = 2,
    step_size: int = 1,
    skip: int = 1,
    max_idx: Optional[int] = None,
    tuple_per_base_img: int = 1,
) -> List[List[int]]:
    idx_tuples: List[List[int]] = []
    if max_idx is None:
        max_idx = -1
    base_idx = img_idx[: max_idx : skip + 1]

    for idx in base_idx:
        for offset in range(tuple_per_base_img):
            imgs_in_list = True
            indices = [idx]
            for tuple_entry in range(1, tuple_length):
                if (idx + (tuple_entry * step_size) + offset) not in img_idx:
                    imgs_in_list = False
                    break
                indices.append(idx + (tuple_entry * step_size) + offset)
            if imgs_in_list:
                idx_tuples.append(indices)
    return idx_tuples
