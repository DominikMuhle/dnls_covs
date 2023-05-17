from enum import Enum

from covpred.dataset.precomp_matches import PreCompMatchesDataset
from covpred.dataset.live_matches import LiveMatchesDataset
from covpred.dataset.visualization import VisDataset
from covpred.dataset.kitti import KITTIPreCompDataset, KITTILiveDataset, KITTIVisualization
from covpred.dataset.euroc import EuRoCPreCompDataset, EuRoCLiveDataset


class Datasets(Enum):
    KITTI = 0
    EuRoC = 1


def get_precomp_dataset(dataset: Datasets) -> type[PreCompMatchesDataset]:
    match dataset:
        case Datasets.KITTI:
            return KITTIPreCompDataset
        case Datasets.EuRoC:
            return EuRoCPreCompDataset
    raise ValueError(f"Did not specify a correct dataset. Got {dataset.name}")


def get_vis_dataset(dataset: Datasets) -> type[VisDataset]:
    match dataset:
        case Datasets.KITTI:
            return KITTIVisualization
    raise ValueError(f"Did not specify a correct dataset. Got {dataset.name}")


def get_live_dataset(dataset: Datasets) -> type[LiveMatchesDataset]:
    match dataset:
        case Datasets.KITTI:
            return KITTILiveDataset
        case Datasets.EuRoC:
            return EuRoCLiveDataset
    raise ValueError(f"Did not specify a correct dataset. Got {dataset.name}")
