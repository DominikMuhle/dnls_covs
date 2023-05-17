from typing import Tuple

import torch
import torch.nn as nn

from covpred.config.model.config import ModelConfig
from covpred.model.output_filter import OutputFilter
from covpred.model.parametrization import BaseParametrization


def extract_covs(
    img_covs: torch.Tensor,  # B*N,H,W,2,2
    host_keypoints: torch.Tensor,  # B,P,M,2
    host_img_idx: torch.Tensor,  # B,P
    target_keypoints: torch.Tensor,  # B,P,M,2
    target_img_idx: torch.Tensor,  # B,P
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = host_img_idx.shape[0]
    num_pairs = host_img_idx.shape[1]
    num_imgs = img_covs.shape[0] // batch_size
    host_imgs = []
    target_imgs = []
    for batch_entry in range(host_img_idx.shape[0]):
        for pair in range(host_img_idx.shape[1]):
            host_imgs.append(img_covs[batch_entry * num_imgs + host_img_idx[batch_entry, pair], ...].to(torch.float32))
            target_imgs.append(
                img_covs[batch_entry * num_imgs + target_img_idx[batch_entry, pair], ...].to(torch.float32)
            )

    host_imgs = torch.stack(host_imgs)  # B*P,H,W,2,2
    target_imgs = torch.stack(target_imgs)  # B*P,H,W,2,2

    norm_host_kps = torch.zeros_like(host_keypoints)
    norm_host_kps[..., 1] = (host_keypoints[..., 1] * 2.0 / (img_covs.shape[1] - 1)) - 1.0
    norm_host_kps[..., 0] = (host_keypoints[..., 0] * 2.0 / (img_covs.shape[2] - 1)) - 1.0

    norm_target_kps = torch.zeros_like(target_keypoints)
    norm_target_kps[..., 1] = (target_keypoints[..., 1] * 2.0 / (img_covs.shape[1] - 1)) - 1.0
    norm_target_kps[..., 0] = (target_keypoints[..., 0] * 2.0 / (img_covs.shape[2] - 1)) - 1.0

    host_covs = torch.nn.functional.grid_sample(
        host_imgs.flatten(-2, -1).permute(0, 3, 1, 2),
        norm_host_kps.flatten(0, 1)[:, None, :, :].to(torch.float32).to(host_imgs.device),
        # mode="bilinear",
        mode="nearest",
        align_corners=True,
    )  # B*P,1,M
    target_covs = torch.nn.functional.grid_sample(
        target_imgs.flatten(-2, -1).permute(0, 3, 1, 2),
        norm_target_kps.flatten(0, 1)[:, None, :, :].to(torch.float32).to(target_imgs.device),
        # mode="bilinear",
        mode="nearest",
        align_corners=True,
    )  # B*P,4,1,M

    return (
        host_covs[:, :, 0, :].permute(0, 2, 1).reshape(host_keypoints.shape[:-1] + (2, 2)), # B*P,M,2,2
        target_covs[:, :, 0, :].permute(0, 2, 1).reshape(target_keypoints.shape[:-1] + (2, 2)), # B*P,M,2,2
    )


def extract_covs_alt(
    img_covs: torch.Tensor,  # B*N,H,W,2,2
    host_keypoints: torch.Tensor,  # B,P,M,2
    host_img_idx: torch.Tensor,  # B,P
    target_keypoints: torch.Tensor,  # B,P,M,2
    target_img_idx: torch.Tensor,  # B,P
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = host_img_idx.shape[0]
    num_pairs = host_img_idx.shape[1]
    num_imgs = img_covs.shape[0] // batch_size
    host_idx = torch.nn.functional.pad(torch.floor(host_keypoints), (2, 0), value=0)  # B,P,M,4
    target_idx = torch.nn.functional.pad(torch.floor(target_keypoints), (2, 0), value=0)  # B,P,M,4
    for i in range(batch_size):
        for j in range(num_pairs):
            host_idx[i, j, ..., 0] = i
            host_idx[i, j, ..., 1] = host_img_idx[i, j]
            target_idx[i, j, ..., 0] = i
            target_idx[i, j, ..., 1] = target_img_idx[i, j]
    host_idx = host_idx.int().to("cpu").reshape(-1, 4).transpose(0, 1).tolist()
    target_idx = target_idx.int().to("cpu").reshape(-1, 4).transpose(0, 1).tolist()

    host_keypoint_covs = img_covs.view(batch_size, num_imgs, img_covs.shape[1], img_covs.shape[2], 2, 2)[
        host_idx[0], host_idx[1], host_idx[3], host_idx[2], :, :
    ].reshape(
        host_keypoints.shape[:-1] + (2, 2)
    )  # B,P,M,2,2
    target_keypoint_covs = img_covs.view(batch_size, num_imgs, img_covs.shape[1], img_covs.shape[2], 2, 2)[
        target_idx[0], target_idx[1], target_idx[3], target_idx[2], :, :
    ].reshape(
        target_keypoints.shape[:-1] + (2, 2)
    )  # B,P,M,2,2
    return host_keypoint_covs, target_keypoint_covs


# TODO: give model config as parameter
class DeepPNEC(nn.Module):
    def __init__(
        self,
        unsc_net: nn.Module,
        output_filter: OutputFilter,
        representation: BaseParametrization,
        model_cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.unsc_net = unsc_net
        self.output_filter = output_filter
        self.representation = representation
        self.cfg = model_cfg

    def forward(
        self,
        images: torch.Tensor,  # B,N,1,H,W
    ) -> torch.Tensor:
        images = images.flatten(0, 1)
        network_output = self.unsc_net(images).permute(0, 2, 3, 1)  # B*N,H,W,3
        return self.representation.back_transform()(
            self.output_filter.filter_output(network_output), False, self.cfg.isotropic_covariances
        )  # B*N,H,W,2,2
