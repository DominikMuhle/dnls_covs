from typing import List, Optional, Tuple

import torch
import theseus as th

from covpred.math.common import angular_diff, cycle_diff


def anchor_error(estimated_poses: th.SE3, anchor_poses: th.SE3, tuple_length: int) -> torch.Tensor:
    return angular_diff(estimated_poses, anchor_poses).reshape(-1, tuple_length).sum(dim=-1)


def consistency_error(estimated_poses: th.SE3, tuple_length: int) -> torch.Tensor:
    return cycle_diff(
        [th.SE3(tensor=pose) for pose in estimated_poses.tensor.reshape(-1, tuple_length, 3, 4).permute(1, 0, 2, 3)]
    )


def angular_error(estimated_poses: th.SE3, ground_truth_poses: th.SE3) -> torch.Tensor:
    return angular_diff(estimated_poses, ground_truth_poses, degree=True)


def self_supervised_loss(
    consistency_error: torch.Tensor,
    loss_function: torch.nn.modules.loss._Loss,
    filter_threshold: float = 1.0,
    additional_errors: Optional[List[Tuple[float, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    filter = consistency_error < filter_threshold
    if filter.sum() == 0:
        filter = torch.ones_like(consistency_error).to(torch.bool)

    loss = loss_function(consistency_error, torch.zeros_like(consistency_error))
    filtered_loss = loss_function(consistency_error[filter], torch.zeros_like(consistency_error[filter]))

    if additional_errors is not None:
        for weight, error in additional_errors:
            loss = loss + weight * loss_function(error, torch.zeros_like(error))
            filtered_loss = filtered_loss + weight * loss_function(error[filter], torch.zeros_like(error[filter]))

    return loss, filtered_loss


def supervised_loss(
    angular_error: torch.Tensor,
    loss_function: torch.nn.modules.loss._Loss,
    filter_threshold: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    filter = angular_error < filter_threshold
    if filter.sum() == 0:
        filter = torch.ones_like(angular_error).to(torch.bool)

    loss = loss_function(angular_error, torch.zeros_like(angular_error))
    filtered_loss = loss_function(angular_error[filter], torch.zeros_like(angular_error[filter]))

    return loss, filtered_loss
