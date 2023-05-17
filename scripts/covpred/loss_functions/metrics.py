from math import sqrt

import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from torch.cuda.amp.autocast_mode import custom_fwd
import theseus as th

from covpred.loss_functions.loss_functions import angular_error, cycle_diff


class CycleError(Metric):
    def __init__(self, tuple_length: int, output_transform=lambda x: x, device: torch.device | str = "cpu"):
        super(CycleError, self).__init__(output_transform=output_transform, device=device)

        self.tuple_length = tuple_length
        self._error = torch.tensor(0.0, device=self._device)
        self._poses = 0

    @reinit__is_reduced
    def reset(self):
        self._error = torch.tensor(0.0, device=self._device)
        self._poses = 0
        super(CycleError, self).reset()

    @reinit__is_reduced
    def update(self, output):
        pred_poses = output[0].detach()

        error = cycle_diff(
            [
                th.SE3(tensor=pose.to(torch.float32))
                for pose in pred_poses.reshape(-1, self.tuple_length, 3, 4).permute(1, 0, 2, 3)
            ]
        )

        self._error += torch.sum(error).to(self._device)
        self._poses += error.shape[0]

    @sync_all_reduce("_error:SUM", "_poses:SUM")
    def compute(self):
        if self._poses == 0:
            return -1.0
            raise NotComputableError("CustomAccuracy must have at least one example before it can be computed.")
        return self._error.item() / (self._poses * (self.tuple_length - 1))


class RotationalError(Metric):
    def __init__(self, output_transform=lambda x: x, device: torch.device | str = "cpu"):
        super(RotationalError, self).__init__(output_transform=output_transform, device=device)

        self._error = torch.tensor(0.0, device=self._device)
        self._poses = 0

    @reinit__is_reduced
    def reset(self):
        self._error = torch.tensor(0.0, device=self._device)
        self._poses = 0
        super(RotationalError, self).reset()

    @reinit__is_reduced
    @custom_fwd(cast_inputs=torch.float32)
    def update(self, output):
        pred_poses, gt_poses = output[0].detach(), output[1].detach()

        error = angular_error(th.SE3(tensor=pred_poses.to(torch.float32)), th.SE3(tensor=gt_poses.to(torch.float32)))

        self._error += torch.sum(error).to(self._device)
        self._poses += error.shape[0]

    @sync_all_reduce("_error:SUM", "_poses:SUM")
    def compute(self):
        if self._poses == 0:
            return -1.0
            raise NotComputableError("CustomAccuracy must have at least one example before it can be computed.")
        return self._error.item() / self._poses


class FilteredRotationalError(Metric):
    def __init__(self, threshold: float, output_transform=lambda x: x, device: torch.device | str = "cpu"):
        super(FilteredRotationalError, self).__init__(output_transform=output_transform, device=device)

        self.threshold = threshold
        self._error = torch.tensor(0.0, device=self._device)
        self._poses = 0

    @reinit__is_reduced
    def reset(self):
        self._error = torch.tensor(0.0, device=self._device)
        self._poses = 0
        super(FilteredRotationalError, self).reset()

    @reinit__is_reduced
    def update(self, output):
        pred_poses, gt_poses = output[0].detach(), output[1].detach()

        error = angular_error(th.SE3(tensor=pred_poses.to(torch.float32)), th.SE3(tensor=gt_poses.to(torch.float32)))
        filter = error < self.threshold

        if filter.sum() > 0:
            self._error += torch.sum(error[filter]).to(self._device)
            self._poses += error[filter].shape[0]

    @sync_all_reduce("_error:SUM", "_poses:SUM")
    def compute(self):
        if self._poses == 0:
            return -1.0
            raise NotComputableError("CustomAccuracy must have at least one example before it can be computed.")
        return self._error.item() / self._poses


class RotationalRMSE(Metric):
    def __init__(self, output_transform=lambda x: x, device: torch.device | str = "cpu"):
        super(RotationalRMSE, self).__init__(output_transform=output_transform, device=device)

        self._error = torch.tensor(0.0, device=self._device)
        self._poses = 0

    @reinit__is_reduced
    def reset(self):
        self._error = torch.tensor(0.0, device=self._device)
        self._poses = 0
        super(RotationalRMSE, self).reset()

    @reinit__is_reduced
    def update(self, output):
        pred_poses, gt_poses = output[0].detach(), output[1].detach()

        error = angular_error(th.SE3(tensor=pred_poses.to(torch.float32)), th.SE3(tensor=gt_poses.to(torch.float32)))

        self._error += torch.sum(torch.square(error)).to(self._device)
        self._poses += error.shape[0]

    @sync_all_reduce("_error:SUM", "_poses:SUM")
    def compute(self):
        if self._poses == 0:
            return -1.0
            raise NotComputableError("CustomAccuracy must have at least one example before it can be computed.")
        return sqrt(self._error.item() / self._poses)
