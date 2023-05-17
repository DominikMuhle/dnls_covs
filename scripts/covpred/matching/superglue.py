from typing import List, Tuple

import torch

from thirdparty.SuperGluePretrainedNetwork.models.matching import Matching

from covpred.matching.base import MatchingFunction, MatchingOutput


def single_matching(
    imgs: torch.Tensor,  # B,N,1,H,W
    host_idx: torch.Tensor,  # B,P
    target_idx: torch.Tensor,  # B,P
    device: torch.device,
    matching: Matching,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    imgs = imgs.to(device)
    mask = []
    img_pts_0 = []
    img_pts_1 = []
    cf = []

    host_imgs = []
    target_imgs = []

    for batch_entry in range(host_idx.shape[0]):
        for pair in range(host_idx.shape[1]):
            host_imgs.append(imgs[batch_entry, host_idx[batch_entry, pair], ...].to(torch.float32))
            target_imgs.append(imgs[batch_entry, target_idx[batch_entry, pair], ...].to(torch.float32))
    pred = matching({"image0": torch.stack(host_imgs), "image1": torch.stack(target_imgs)})
    for i in range(len(host_imgs)):
        # pred_i = {k: v[i].cpu().detach().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred["keypoints0"][i], pred["keypoints1"][i]
        matches, conf = pred["matches0"][i], pred["matching_scores0"][i]
        valid = matches > -1
        num_points = valid.shape[0]
        if valid.sum() < 10:
            print(f"[WARNING]: Coudn't find enough matches")
        mask.append(torch.ones_like(kpts0[:num_points][valid])[..., 0])
        img_pts_0.append(kpts0[:num_points][valid])
        img_pts_1.append(kpts1[matches[valid]])
        cf.append(conf[valid])
        # mask.append(torch.ones_like(torch.from_numpy(kpts0[valid]))[..., 0])
        # img_pts_0.append(torch.from_numpy(kpts0[valid]))
        # img_pts_1.append(torch.from_numpy(kpts1[matches[valid]]))
        # cf.append(torch.from_numpy(conf[valid]))

    return img_pts_0, img_pts_1, cf, mask


def SuperGlue(matching: Matching, half: bool = False) -> MatchingFunction:
    def _matching(
        images: torch.Tensor,  # B,N,1,H,W
        K_inv: torch.Tensor,  # B,N,3,3
        host_idx: torch.Tensor,  # B,P
        target_idx: torch.Tensor,  # B,P
        device: torch.device,
    ) -> MatchingOutput:
        if half:
            bs_half = images.shape[0] // 2
            matching_0 = single_matching(images[:bs_half], host_idx[:bs_half], target_idx[:bs_half], device, matching)
            matching_1 = single_matching(images[bs_half:], host_idx[bs_half:], target_idx[bs_half:], device, matching)
            img_pts_0 = matching_0[0] + matching_1[0]
            img_pts_1 = matching_0[1] + matching_1[1]
            cf = matching_0[2] + matching_1[2]
            mask = matching_0[3] + matching_1[3]
        else:
            img_pts_0, img_pts_1, cf, mask = single_matching(images, host_idx, target_idx, device, matching)

        return (
            torch.nn.utils.rnn.pad_sequence(img_pts_0, True, 0.0),
            torch.nn.utils.rnn.pad_sequence(img_pts_1, True, 0.0),
            torch.nn.utils.rnn.pad_sequence(cf, True, 0.0),
            torch.nn.utils.rnn.pad_sequence(mask, True, 0.0),
        )

    return _matching
