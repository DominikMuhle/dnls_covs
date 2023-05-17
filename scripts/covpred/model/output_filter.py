from abc import ABC
from enum import Enum
from typing import Callable, Tuple

import torch

FilterFunction = Callable[[torch.Tensor], torch.Tensor]


class OutputFilter(ABC):
    def __init__(
        self,
        filter1: Tuple[FilterFunction, FilterFunction],
        filter2: Tuple[FilterFunction, FilterFunction],
        filter3: Tuple[FilterFunction, FilterFunction],
    ) -> None:
        super().__init__()
        self.filter1 = filter1[0]
        self.filter2 = filter2[0]
        self.filter3 = filter3[0]

        self.inv_filter1 = filter1[1]
        self.inv_filter2 = filter2[1]
        self.inv_filter3 = filter3[1]

    def filter_output(self, output: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.filter1(output[..., 0]), self.filter2(output[..., 1]), self.filter3(output[..., 2])], dim=-1
        )

    def inv_filter_output(self, output: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.inv_filter1(output[..., 0]), self.inv_filter2(output[..., 1]), self.inv_filter3(output[..., 2])],
            dim=-1,
        )


def sigmoid_filter(min: float = 0, max: float = 1) -> FilterFunction:
    def _sigmoid_filter(tensor: torch.Tensor) -> torch.Tensor:
        return (max - min) * torch.sigmoid(tensor) + min

    return _sigmoid_filter


def inv_sigmoid_filter(min: float = 0, max: float = 1) -> FilterFunction:
    def _inv_sigmoid_filter(tensor: torch.Tensor) -> torch.Tensor:
        return -torch.log(((max - min) / (tensor - min)) - 1)
        # return torch.log((1.0 / (tensor - min)) - 1.0)) / (max - min)

    return _inv_sigmoid_filter


def lukas_filter() -> FilterFunction:
    def _lukas_filter(tensor: torch.Tensor) -> torch.Tensor:
        output = torch.ones_like(tensor)
        output[tensor < 0] = (1 / (1 - tensor[tensor < 0])).to(output.dtype)
        output[tensor >= 0] = (1 + tensor[tensor >= 0]).to(output.dtype)
        return output

    return _lukas_filter


def inv_lukas_filter() -> FilterFunction:
    def _inv_lukas_filter(tensor: torch.Tensor) -> torch.Tensor:
        output = torch.ones_like(tensor)
        output[tensor >= 1] = tensor[tensor >= 1] - 1
        output[tensor < 1] = 1 - (1 / tensor[tensor < 1])
        return output

    return _inv_lukas_filter


def exp_filter() -> FilterFunction:
    def _exp_filter(tensor: torch.Tensor) -> torch.Tensor:
        return torch.exp(tensor)

    return _exp_filter


def inv_exp_filter() -> FilterFunction:
    def _inv_exp_filter(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor)

    return _inv_exp_filter


def relu_filter() -> FilterFunction:
    def _relu_filter(tensor: torch.Tensor) -> torch.Tensor:
        return torch.relu(tensor)

    return _relu_filter


def inv_relu_filter() -> FilterFunction:
    # no real inverse possible
    def _inv_relu_filter(tensor: torch.Tensor) -> torch.Tensor:
        return torch.relu(tensor)

    return _inv_relu_filter


def no_filter() -> FilterFunction:
    def _no_filter(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    return _no_filter


def inv_no_filter() -> FilterFunction:
    def _inv_no_filter(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    return _inv_no_filter


class Filters(Enum):
    no = 0
    sigmoid = 1
    lukas = 2
    exp = 3
    relu = 4


def get_filter(filter: Filters, **kwargs) -> Tuple[FilterFunction, FilterFunction]:
    if filter == Filters.no:
        return no_filter(), inv_no_filter()
    if filter == Filters.sigmoid:
        return sigmoid_filter(**kwargs), inv_sigmoid_filter(**kwargs)
    if filter == Filters.lukas:
        return lukas_filter(), inv_lukas_filter()
    if filter == Filters.exp:
        return exp_filter(), inv_exp_filter()
    if filter == Filters.relu:
        return relu_filter(), inv_relu_filter()
    return no_filter(), inv_no_filter()
