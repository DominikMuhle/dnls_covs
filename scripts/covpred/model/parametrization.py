from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import numpy as np

from covpred.model import transforms

Transform = Callable[[torch.Tensor], torch.Tensor]
BackTransform = Callable[[torch.Tensor, bool, bool], torch.Tensor]
ImageTransform = Callable[[torch.Tensor], np.ndarray]


class BaseParametrization(ABC):
    def names(self) -> Tuple[str, str, str]:
        return ("", "", "")

    @abstractmethod
    def transform(self) -> Transform:
        """Returns a transform function for covariance matrices that produces a
        reprensentation that can be used for learning.

        Returns:
            UncertaintyTransform: transformation function for a
                                                set of covariances
        """
        pass

    @abstractmethod
    def back_transform(self) -> BackTransform:
        """Returns a transform function that converts a representation to
        covariance matrices.

        Returns:
            UncertaintyTransform: transformation function for a
            set of representations.
        """

    @abstractmethod
    def to_img(self) -> ImageTransform:
        """Returns a transform function that converts a representation to
        and image.

        Returns:
            UncertaintyTransform: transformation function for a
            set of representations.
        """


class ScaleRotationAnsitropyRepresentation(BaseParametrization):
    def names(self) -> Tuple[str, str, str]:
        return ("scale", "alpha", "beta")

    def __init__(self, scale_limits: Tuple[Optional[float], Optional[float]] = (None, None)) -> None:
        super().__init__()
        self.scale_limits = scale_limits

    def __str__(self) -> str:
        return "ScaleRotationAnsiotropy"

    def transform(self) -> Transform:
        return partial(transforms.parameters_from_covs, scale_limits=self.scale_limits)

    def back_transform(self) -> BackTransform:
        return transforms.covs_from_parameters

    def to_img(self) -> ImageTransform:
        return transforms.sab_to_image


class MatrixEntriesRepresentation(BaseParametrization):
    def names(self) -> Tuple[str, str, str]:
        return ("cov_11", "cov_12", "cov_22")

    def __str__(self) -> str:
        return "MatrixEntries"

    def transform(self) -> Transform:
        return transforms.entries_from_covs

    def back_transform(self) -> BackTransform:
        return transforms.covs_from_entries

    def to_img(self) -> ImageTransform:
        return transforms.entries_to_image


class InverseMatrixEntriesRepresentation(BaseParametrization):
    def names(self) -> Tuple[str, str, str]:
        return ("H_11", "H_12", "H_22")

    def __str__(self) -> str:
        return "InvMatrixEntries"

    def transform(self) -> Transform:
        return transforms.entries_from_inv_covs

    def back_transform(self) -> BackTransform:
        return transforms.covs_from_inverse_entries

    def to_img(self) -> ImageTransform:
        return transforms.inverse_entries_to_image


class CovarianceParameterization(Enum):
    sab = 1
    entries = 2
    inv_entries = 3


def get_parametrization(representation: CovarianceParameterization, **kwargs) -> BaseParametrization:
    if representation == CovarianceParameterization.sab:
        return ScaleRotationAnsitropyRepresentation(**kwargs)
    if representation == CovarianceParameterization.entries:
        return MatrixEntriesRepresentation()
    if representation == CovarianceParameterization.inv_entries:
        return InverseMatrixEntriesRepresentation()
    return MatrixEntriesRepresentation()
