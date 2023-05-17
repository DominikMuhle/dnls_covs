from enum import Enum
import logging
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from skimage import io
import theseus as th
from theseus.optimizer.nonlinear.nonlinear_optimizer import NonlinearOptimizerInfo
from torch._six import string_classes
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from covpred.config.model.config import ModelOutputConfig
from covpred.model.output_filter import Filters, OutputFilter, get_filter
from covpred.math.projections import linear


def complete_pnec_covariances(
    host_covs: torch.Tensor | None, target_covs: torch.Tensor | None
) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: check if handled by match
    # TODO: check if overloading works
    if host_covs is None and target_covs is None:
        raise ValueError("Expected at least one covariance tensor to be not None.")

    if host_covs is None and target_covs is not None:
        host_covs = torch.zeros_like(target_covs)
        return host_covs, target_covs

    if host_covs is not None and target_covs is None:
        target_covs = torch.zeros_like(host_covs)
        return host_covs, target_covs

    if host_covs is not None and target_covs is not None:
        return host_covs, target_covs

    # This is never reached but needed for type checking
    return torch.zeros(1, 3, 3), torch.zeros(1, 3, 3)


class OptInfo(Metric):
    required_output_keys = ("converged_iter", "last_err", "err_history")

    def __init__(self, output_transform=lambda x: x, device: torch.device | str = "cpu"):
        super(OptInfo, self).__init__(output_transform=output_transform, device=device)

        self._diverged = 0
        self._not_converged = 0
        self._num_optimizations = 0
        self._max_conv_iter = 0
        self._bad_gn_step = 0
        self._state_history: Dict[str, List[torch.Tensor]] | None = None  # variable name to state history

    @reinit__is_reduced
    def reset(self):
        self._diverged = 0
        self._not_converged = 0
        self._num_optimizations = 0
        self._max_conv_iter = 0
        self._bad_gn_step = 0
        super(OptInfo, self).reset()

    @reinit__is_reduced
    def update(self, output):
        converged_iter, last_err, err_history = output
        num_optimizations = last_err.size()[0]

        diverged = 0
        bad_gn_step = 0
        if err_history is not None:
            diverged = (last_err.to("cpu") - err_history[..., 0] > 0).sum().item()
            convergence_iter = converged_iter
            convergence_iter[convergence_iter == -1] = -2
            semi_last_err = err_history[range(convergence_iter.shape[0]), convergence_iter.tolist()]
            bad_gn_step = (last_err.to("cpu") / semi_last_err > 1.0001).sum().item()

        not_coverged = (converged_iter == -1).sum().item()
        max_conv_iter = converged_iter.max().item()

        self._num_optimizations += num_optimizations
        self._diverged += diverged
        self._bad_gn_step += bad_gn_step
        self._not_converged += not_coverged
        self._max_conv_iter = max(self._max_conv_iter, max_conv_iter)

    @sync_all_reduce(
        "_diverged:SUM", "_not_converged:SUM", "_bad_gn_step:SUM", "_num_optimizations:SUM", "_max_conv_iter:MAX"
    )
    def compute(self):
        return self._num_optimizations, self._max_conv_iter, self._not_converged, self._diverged, self._bad_gn_step


model_logger = logging.getLogger("Model")


def model_info(unsc_net: torch.nn.Module) -> None:
    # info
    param_size = 0
    param_nelement = 0
    for param in unsc_net.parameters():
        param_nelement += param.nelement()
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    buffer_nelement = 0
    for buffer in unsc_net.buffers():
        buffer_nelement += buffer.nelement()
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    model_logger.info(f"Model has {param_nelement + buffer_nelement} parameters in total.")
    model_logger.info(f"Model has a size of {size_all_mb:.3} MB.")


def get_largest_gpus() -> List[int]:
    ids: List[int] = []
    largest_memory = 0
    for device_id in range(torch.cuda.device_count()):
        device_memory = torch.cuda.get_device_properties(device_id).total_memory
        if device_memory > largest_memory:
            largest_memory = device_memory
            ids = []
        if device_memory == largest_memory:
            ids.append(device_id)
    return ids


class DummySummaryWriter:
    def __init__(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self


def img2tensor(img: np.ndarray, precision: torch.dtype) -> torch.Tensor:
    if img.ndim == 3:
        return torch.from_numpy(img / 255.0).permute(2, 0, 1).to(precision)
    return torch.from_numpy(img / 255.0).to(precision)[None]


def get_img_path(dataset_dir: Path, img_dir: str, seq: str, idx: int) -> Path:
    image_name = f"{str(idx).zfill(6)}.png"
    return dataset_dir.joinpath(seq, img_dir, image_name)


def load_img(path: Path, precision: torch.dtype = torch.float64) -> torch.Tensor:
    return img2tensor(io.imread(path.resolve()), precision)


class Initializations(Enum):
    IDENTITY = 1
    GROUND_TRUTH = 2
    RANDOM_PERTUBATION = 3
    RANDOM = 4


def get_initialization(
    initialization: Initializations = Initializations.IDENTITY,
    gt_poses: Optional[th.SE3] = None,
    num_poses: Optional[int] = None,
    max_angle: float = 1.0,  # degree
    max_translation: float = 1.0,
) -> th.SE3:
    def _identity_init() -> th.SE3:
        nonlocal num_poses
        if num_poses is None:
            num_poses = 1
        return th.SE3(
            tensor=torch.nn.functional.pad(
                torch.eye(3)[None, ...].expand(num_poses, -1, -1),
                (0, 1),
                value=0.0,
            )
        )

    def _gt_init() -> th.SE3:
        if gt_poses is None:
            print("WARNING: No ground truth poses, falling back on identity initialization.")
            return _identity_init()
        return gt_poses

    def _random_per_init() -> th.SE3:
        if gt_poses is None:
            print("WARNING: No ground truth poses, falling back on identity initialization.")
            return _identity_init()
        rot_offset_so3 = (
            torch.randn((gt_poses.tensor.shape[0], 3), device=gt_poses.tensor.device, dtype=gt_poses.dtype)
            * max_angle
            * torch.pi
            / 180.0
        )
        trans_offset = (
            torch.randn((gt_poses.tensor.shape[0], 3), device=gt_poses.tensor.device, dtype=gt_poses.dtype)
            * max_translation
        )
        offset = th.SE3.exp_map(torch.concat([trans_offset, rot_offset_so3], dim=-1))
        # casting needed to get rid of typing error
        return th.SE3(tensor=offset.compose(gt_poses).tensor)

    def _random_init() -> th.SE3:
        nonlocal num_poses
        if num_poses is None:
            num_poses = 1

        rot_offset_so3 = torch.randn((num_poses, 3)) * max_angle * torch.pi / 180.0
        trans_offset = torch.randn((num_poses, 3)) * max_translation

        return th.SE3().exp_map(torch.concat([trans_offset, rot_offset_so3], dim=-1))

    if initialization == Initializations.IDENTITY:
        return _identity_init()
    if initialization == Initializations.GROUND_TRUTH:
        return _gt_init()
    if initialization == Initializations.RANDOM_PERTUBATION:
        return _random_per_init()
    if initialization == Initializations.RANDOM:
        return _random_init()
    return _identity_init()


def get_entries_in_batch(tensor: torch.Tensor, indices: torch.Tensor, batch_size: int) -> torch.Tensor:
    return tensor[[i for i in range(batch_size) for _ in range(indices.shape[1])], indices.flatten(0, 1).tolist(), ...]


def get_rel_poses(
    absolut_poses: torch.Tensor, host_idx: torch.Tensor, target_idx: torch.Tensor, batch_size: int
) -> th.SE3:
    # casting needed for getting rid of typing error
    return th.SE3(
        tensor=th.SE3(tensor=get_entries_in_batch(absolut_poses, host_idx, batch_size))
        .between(th.SE3(tensor=get_entries_in_batch(absolut_poses, target_idx, batch_size)))
        .tensor
    )


def rel_2_abs_poses(rel_poses: torch.Tensor) -> torch.Tensor:
    abs_poses = [th.SE3(dtype=rel_poses.dtype).tensor]
    for pose in rel_poses:
        abs_poses.append(th.SE3(tensor=abs_poses[-1]).compose(th.SE3(tensor=pose[None])).tensor)
    return torch.concat(abs_poses, dim=0)


class TranslationMode(Enum):
    UNITLENGTH = 1
    GTLENGTH = 2
    GTFULL = 3


def translation_scale(poses: torch.Tensor, gt_poses: torch.Tensor, mode: TranslationMode) -> torch.Tensor:
    if mode == TranslationMode.GTLENGTH:
        if torch.dot(poses[..., :3, 3], gt_poses[..., :3, 3]) < 0:
            poses[..., :3, 3] = -1.0 * poses[..., :3, 3]
        length_gt_poses = torch.linalg.norm(gt_poses[..., :3, 3])
        poses[..., :3, 3] = poses[..., :3, 3] * (length_gt_poses / torch.linalg.norm(poses[..., :3, 3]))
    if mode == TranslationMode.GTFULL:
        poses[..., :3, 3] = gt_poses[..., :3, 3]
    return poses


def to_3d_cov(covariances: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    if covariances.shape[-2:] == (3, 3):
        return covariances
    if covariances.shape[-2:] == (2, 2):
        cov_3d = torch.nn.functional.pad(covariances, (0, 1, 0, 1), value=value)
        return cov_3d
    raise ValueError(f"Expected covariance of shape (..., 2, 2) or (..., 3, 3), got {covariances.shape}")


def to_3d_point(points: torch.Tensor, value: float = 1.0) -> torch.Tensor:
    if points.shape[-1] == 3:
        return points
    if points.shape[-1] == 2:
        return torch.nn.functional.pad(points, (0, 1), value=value)
    raise ValueError(f"Expected points of shape (..., 2) or (..., 3), got {points.shape}")


def subsample_points(
    tensor: torch.Tensor,
    subsampling: Union[int, List[int]] = 1,
) -> torch.Tensor:
    if isinstance(subsampling, int):
        return tensor[::subsampling]
    return tensor[subsampling]


def create_bearing_vectors(
    img_kps: torch.Tensor,
    covs: torch.Tensor,
    K_inv: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bvs, bvs_covs = linear(
        to_3d_point(img_kps),
        K_inv.type_as(img_kps),
        to_3d_cov(covs).type_as(img_kps),
    )
    if mask is not None:
        bvs = bvs * mask[..., None]

    return bvs, bvs_covs


def load_output_filter(filter_cfg: ModelOutputConfig) -> OutputFilter:
    return OutputFilter(
        get_filter(Filters[filter_cfg.filter1.name], **(filter_cfg.filter1.args or {})),
        get_filter(Filters[filter_cfg.filter2.name], **(filter_cfg.filter2.args or {})),
        get_filter(Filters[filter_cfg.filter3.name], **(filter_cfg.filter3.args or {})),
    )


def get_percentile(tensor: torch.Tensor, percentile: float) -> torch.Tensor:
    return torch.sort(tensor)[0][: (ceil(percentile * tensor.shape[0]))]


def filter_keypoints(
    host_keypoints: torch.Tensor, target_keypoints: torch.Tensor, masks: torch.Tensor, img_size: Tuple[int, int]
) -> None:
    host_in_bounds = points_in_img(host_keypoints, img_size)
    target_in_bounds = points_in_img(target_keypoints, img_size)

    out_bounds = torch.logical_not(torch.logical_and(host_in_bounds, target_in_bounds))
    host_keypoints[out_bounds] = 0
    target_keypoints[out_bounds] = 0
    masks[out_bounds] = 0


def flip_keypoints(keypoints: torch.Tensor, width: int):
    keypoints[..., 0] = width - keypoints[..., 0]


def write_losses(
    writer: Union[SummaryWriter, DummySummaryWriter],
    losses: Dict[str, torch.Tensor],
    base_rotational_error: Dict[str, torch.Tensor],
    name: str,
    epoch: int,
):
    percentile = 0.95

    def write_to_tensorboard(loss: torch.Tensor, loss_name: str, step: int):
        writer.add_scalar(f"{loss_name}/{name}_mean", loss.mean(), step)
        if loss_name == "rotational error":
            # if loss_name == "error rotational":
            writer.add_scalar(f"{loss_name}/{name}_median", loss.median(), step)
            writer.add_scalar(
                f"{loss_name}/{name}_{percentile * 100}%mean", get_percentile(loss, percentile).mean(), step
            )
            writer.add_scalar(f"{loss_name}/{name}_rmse", torch.sqrt(torch.square(loss).mean()), step)
            # writer.add_histogram(f"{loss_name}-{name}", loss, step)
            # writer.add_histogram(f"Lower-{loss_name}-{name}", get_smaller_than(loss, hist_limit), step)
            nec_loss = base_rotational_error.get("NEC", None)
            if nec_loss is not None:
                better_than = (nec_loss - loss > 0).sum()
                writer.add_scalar(f"better than nec/{name}", better_than / loss.shape[0], step)
            nec_ls_loss = base_rotational_error.get("NEC-LS", None)
            if nec_ls_loss is not None:
                better_than = (nec_ls_loss - loss > 0).sum()
                writer.add_scalar(f"better than nec_ls/{name}", better_than / loss.shape[0], step)
            klt_loss = base_rotational_error.get("KLT", None)
            if klt_loss is not None:
                better_than = (klt_loss - loss > 0).sum()
                writer.add_scalar(f"better than klt/{name}", better_than / loss.shape[0], step)

    for loss_name, loss in losses.items():
        write_to_tensorboard(loss, loss_name, epoch)


def write_infos(writer: Union[SummaryWriter, DummySummaryWriter], info: NonlinearOptimizerInfo, name: str, epoch: int):
    num_optimizations = info.last_err.size()[0]
    if info.err_history is not None:
        diverged = (info.last_err.to("cpu") - info.err_history[..., 0] > 0).sum()
        if diverged:
            print(f"[Warning]: Out of {num_optimizations} optimizations {diverged.item()} diverged")

    # convergence
    has_converged_iter = info.converged_iter[info.converged_iter != -1]
    writer.add_histogram(f"convergence iterations/{name}", has_converged_iter, epoch)

    # bad last step
    convergence_iter = info.converged_iter
    convergence_iter[convergence_iter == -1] = -2
    if info.err_history is not None:
        semi_last_err = info.err_history[range(convergence_iter.shape[0]), convergence_iter.tolist()]
        bad_last_step = (info.last_err.to("cpu") / semi_last_err > 1.0001).sum()
        if bad_last_step:
            print(
                f"[Warning]: Out of {num_optimizations} optimizations {bad_last_step.item()} had a bad last gauss newton step"
            )

    if info.err_history is not None:
        writer.add_histogram(f"OOM Cost Beginning/{name}", torch.log10(info.err_history[..., 0]), epoch)
    writer.add_histogram(f"OOM Cost End/{name}", torch.log10(info.last_err), epoch)


def points_in_img(points: torch.Tensor, img_size: Tuple[int, int], margin: Tuple[int, int] = (0, 0)) -> torch.Tensor:
    y_in_bounds = torch.logical_and(points[..., 1] >= 0 + margin[1], points[..., 1] < img_size[0] - margin[1])
    x_in_bounds = torch.logical_and(points[..., 0] >= 0 + margin[0], points[..., 0] < img_size[1] - margin[0])
    return torch.logical_and(x_in_bounds, y_in_bounds)


from torch.utils.data import get_worker_info
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from collections.abc import Mapping, Sequence


# TODO: rework with new collate adaption
def custom_collate(batch):
    r"""
    Function that takes in a batch of data and puts the elements within the batch
    into a tensor with stacking them in the first dimension with length N - batch size * N.
    The exact output type can be a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`,
    a Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as a custom function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.
    Here is the general input type (based on the type of the element within the batch) to output type mapping:
    * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
    * NumPy Arrays -> :class:`torch.Tensor`
    * `float` -> :class:`torch.Tensor`
    * `int` -> :class:`torch.Tensor`
    * `str` -> `str` (unchanged)
    * `bytes` -> `bytes` (unchanged)
    * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
    * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
    * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
    Args:
        batch: a single batch to be collated
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, Mapping):
        try:
            return elem_type({key: custom_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [custom_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([custom_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
