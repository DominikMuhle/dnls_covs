from abc import abstractmethod
from math import inf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
from torch.utils.data import Dataset
import theseus as th

from covpred.common import Initializations, get_initialization, load_img, to_3d_point
from covpred.config.dataset.config import DatasetConfig
from covpred.config.pose_estimation.config import NECConfig
from covpred.dataset.common import ImgData, ImgPair, ImgTupleWithMatches, create_idx_tuples
from covpred.dataset.augmentations.keypoints import add_jitter
import covpred.matching as matching
from covpred.io.matches import load_matches, write_matches
from covpred.math.projections import linear
from covpred.model.optimization.eigenvalue import nec


PreCompData = Tuple[
    torch.Tensor,  # images
    torch.Tensor,  # K_matrix
    torch.Tensor,  # poses
    torch.Tensor,  # host_kps
    torch.Tensor,  # host_img_idx
    torch.Tensor,  # target_kps
    torch.Tensor,  # target_img_idx
    torch.Tensor,  # masks
    torch.Tensor,  # confidences
    Optional[torch.Tensor],  # init_poses
    Optional[torch.Tensor],  # nec_poses
]

dataset_logger = logging.getLogger("Dataset")


class PreCompMatchesDataset(Dataset):
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
        if split not in dataset_config.directories.sequences.keys():
            raise ValueError(
                f"Split not available in the given sequences orderings. Available splits: {dataset_config.directories.sequences.keys()}. Name provided: {split}"
            )

        self.dataset_config = dataset_config
        self.nec_config = nec_config
        self.split = split
        self.device = device
        self.override_matches = override_matches
        self.random_init = random_init
        self.matching_fn = matching_fn
        sequences = self.dataset_config.directories.sequences[split]
        if isinstance(sequences, str):
            sequences = [sequences]
        self.sequences = sequences
        self.img_precision = torch.float32
        self.min_size = (inf, inf)

        self.img_tuples = self.create_img_tuple()

    def __len__(self) -> int:
        return len(self.img_tuples)

    def get_sequence_indices(self) -> Dict[str, List[int]]:
        sequence_indices: Dict[str, List[int]] = {}
        for idx, img_tuple in enumerate(self.img_tuples):
            sequence = img_tuple.images[0].sequence
            if sequence not in sequence_indices.keys():
                sequence_indices[sequence] = []
            sequence_indices[sequence].append(idx)
        return sequence_indices

    @abstractmethod
    def get_img_path(self, sequence: str, idx: int) -> Path:
        pass

    @abstractmethod
    def load_sequence(self, sequence_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        pass

    def create_img_tuple(self) -> List[ImgTupleWithMatches]:
        img_tuples = []
        sequences = self.dataset_config.directories.sequences[self.split]

        for sequence in sequences:
            # if type(sequence) == int:
            #     sequence = str(sequence)
            seq_dir = Path(self.dataset_config.directories.dataset).joinpath(sequence)
            sequence_matches_path = Path(self.dataset_config.directories.matches).joinpath(
                self.dataset_config.name, self.dataset_config.matching.algorithm, sequence
            )

            test_image = load_img(self.get_img_path(sequence, 0))
            self.min_size = (min(self.min_size[0], test_image.shape[1]), min(self.min_size[1], test_image.shape[2]))

            poses, K_matrix, indices = self.load_sequence(seq_dir)

            idx_tuples = create_idx_tuples(
                indices,
                self.dataset_config.sweep.tuple_length,
                self.dataset_config.sweep.step_size,
                self.dataset_config.sweep.skip,
                self.dataset_config.sweep.max_idx,
                self.dataset_config.sweep.tuple_per_base_img,
            )
            dataset_logger.info(f'Looking for images in "{seq_dir}".')

            matches_present = 0
            new_matches = 0
            for idx_tuple in idx_tuples:
                images: List[ImgData] = []
                pairs: Dict[Tuple[int, int], ImgPair] = {}
                for img_idx in idx_tuple:
                    images.append(ImgData(sequence, img_idx, K_matrix, th.SE3(tensor=poses[img_idx][None])))
                for tuple_idx_host, host_idx in enumerate(idx_tuple):
                    if tuple_idx_host + 1 == 2 and len(idx_tuple) == 2:
                        break

                    if tuple_idx_host + 1 == len(idx_tuple):
                        tuple_idx_target = 0
                    else:
                        tuple_idx_target = tuple_idx_host + 1
                    target_idx = idx_tuple[tuple_idx_target]

                    matches = None
                    if not self.override_matches:
                        matches = load_matches(sequence_matches_path, host_idx, target_idx)

                    if matches is None:
                        host_img = load_img(self.get_img_path(sequence, host_idx), self.img_precision)  # 1,H,W
                        target_img = load_img(self.get_img_path(sequence, target_idx), self.img_precision)  # 1,H,W
                        imgs = torch.stack([host_img, target_img])[None]  # 1,2,1,H,W

                        K_inv = torch.linalg.inv(K_matrix[None].expand(2, -1, -1))[None]  # 1,2,3,3
                        matches = self.matching_fn(imgs, K_inv, torch.tensor([[0]]), torch.tensor([[1]]), self.device)

                        write_matches(sequence_matches_path, host_idx, target_idx, matches)
                        new_matches += 1
                    else:
                        matches_present += 1

                    if matches[2].max() != matches[2].min():
                        sort_indices = torch.argsort(matches[2], descending=True)
                        matches = (
                            matches[0][sort_indices],
                            matches[1][sort_indices],
                            matches[2][sort_indices],
                            matches[3][sort_indices],
                        )

                    host_keypoints = matches[0][: self.dataset_config.matching.max_points, :]
                    target_keypoints = matches[1][: self.dataset_config.matching.max_points, :]
                    confidence = matches[2][: self.dataset_config.matching.max_points]
                    mask = matches[3][: self.dataset_config.matching.max_points]

                    if self.random_init:
                        rel_pose = th.SE3(
                            tensor=images[tuple_idx_host].pose.between(images[tuple_idx_target].pose).tensor
                        )
                        init_pose = get_initialization(
                            Initializations.RANDOM_PERTUBATION, rel_pose, max_angle=0.05, max_translation=0.05
                        )

                        # TODO: adapt to other projections
                        bvs_0 = linear(
                            to_3d_point(host_keypoints), torch.linalg.inv(K_matrix)[None].type_as(host_keypoints)
                        )[0]
                        bvs_1 = linear(
                            to_3d_point(target_keypoints), torch.linalg.inv(K_matrix)[None].type_as(host_keypoints)
                        )[0]

                        nec_pose, inliers = nec(bvs_0, bvs_1, init_pose, self.nec_config)
                    else:
                        init_pose, nec_pose = None, None

                    if False:
                        host_keypoints, target_keypoints = self.augment_keypoints(host_keypoints, target_keypoints)

                    pairs[(tuple_idx_host, tuple_idx_target)] = ImgPair(
                        host_keypoints, target_keypoints, mask, confidence, init_pose, nec_pose
                    )
                img_tuples.append(ImgTupleWithMatches(images, pairs))
            dataset_logger.info(
                f"For {len(idx_tuples)} tuples found {matches_present} pairs with matches, created {new_matches} new ones."
            )

        return img_tuples

    def augment_keypoints(
        self, host_keypoints: torch.Tensor, target_keypoints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Make better, with different augmentations in arbitrary order
        # if self.dataset_config.jitter_std > 0.0:
        #     host_keypoints = add_jitter(host_keypoints, self.dataset_config.jitter_std)
        #     target_keypoints = add_jitter(target_keypoints, self.dataset_config.jitter_std)

        return host_keypoints, target_keypoints

    def __getitem__(self, index) -> PreCompData:
        if torch.is_tensor(index):
            index = index.tolist()

        sequence = self.img_tuples[index].images[0].sequence
        K_matrix = []
        images: List[torch.Tensor] = []
        poses: List[torch.Tensor] = []
        for img in self.img_tuples[index].images:
            images.append(
                load_img(self.get_img_path(sequence, img.idx), self.img_precision)[
                    ...,
                    : self.min_size[0],
                    : self.min_size[1],
                ]
            )
            poses.append(img.pose.tensor)
            K_matrix.append(img.K_matrix)
        host_keypoints = []
        host_idx = []
        target_keypoints = []
        target_idx = []
        masks: List[torch.Tensor] = []
        confidences: List[torch.Tensor] = []
        rel_poses = []
        init_poses = []
        nec_poses = []
        for (tuple_idx_host, tuple_idx_target), img_pair in self.img_tuples[index].pairs.items():
            host_keypoints.append(img_pair.host_keypoints)
            host_idx.append(tuple_idx_host)
            target_keypoints.append(img_pair.target_keypoints)
            target_idx.append(tuple_idx_target)
            masks.append(img_pair.mask)
            confidences.append(img_pair.confidence)
            rel_poses.append(
                th.SE3(tensor=poses[tuple_idx_host]).between(th.SE3(tensor=poses[tuple_idx_target])).tensor
            )
            if img_pair.init_pose is not None:
                init_poses.append(img_pair.init_pose.tensor)
            if img_pair.nec_pose is not None:
                nec_poses.append(img_pair.nec_pose.tensor)

        if self.random_init:
            init_poses = torch.concat(init_poses, dim=0)
            nec_poses = torch.concat(nec_poses, dim=0)
        else:
            init_poses, nec_poses = torch.zeros_like(torch.concat(poses, dim=0)), torch.zeros_like(
                torch.concat(poses, dim=0)
            )

        return (
            torch.stack(images, dim=0),
            torch.linalg.inv(torch.stack(K_matrix, dim=0)),
            torch.concat(rel_poses, dim=0),
            torch.nn.utils.rnn.pad_sequence(host_keypoints, batch_first=False),
            torch.tensor(host_idx, dtype=torch.int),
            torch.nn.utils.rnn.pad_sequence(target_keypoints, batch_first=False),
            torch.tensor(target_idx, dtype=torch.int),
            torch.nn.utils.rnn.pad_sequence(masks, batch_first=False),
            torch.nn.utils.rnn.pad_sequence(confidences, batch_first=False),
            init_poses,
            nec_poses,
        )
