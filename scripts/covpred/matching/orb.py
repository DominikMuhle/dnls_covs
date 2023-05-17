from typing import Callable, Tuple
import numpy as np

import torch
import cv2 as cv


def ORB(
    images: torch.Tensor, K_inv: torch.Tensor, host_idx: torch.Tensor, target_idx: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    host_kps = []
    target_kps = []
    masks = []
    for i in range(host_idx.shape[0]):  # batch entries
        for j in range(host_idx.shape[1]):  # pairs
            kp1, des1 = orb.detectAndCompute(
                (images[i, host_idx[i, j]][0] * 255.0).to("cpu").detach().numpy().astype(np.uint8), None
            )
            kp2, des2 = orb.detectAndCompute(
                (images[i, target_idx[i, j]][0] * 255.0).to("cpu").detach().numpy().astype(np.uint8), None
            )
            matches = bf.match(des1, des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)

            host_kps.append(torch.from_numpy(np.array([kp1[match.queryIdx].pt for match in matches])))
            target_kps.append(torch.from_numpy(np.array([kp2[match.trainIdx].pt for match in matches])))
            masks.append(torch.ones_like(host_kps[-1])[..., 0])

    host_kps = torch.nn.utils.rnn.pad_sequence(host_kps, True, 0.0)
    target_kps = torch.nn.utils.rnn.pad_sequence(target_kps, True, 0.0)
    masks = torch.nn.utils.rnn.pad_sequence(masks, True, 0.0)
    flip = False
    if flip:
        host_kps = host_kps.flip(-1)
        target_kps = target_kps.flip(-1)

    return host_kps.to(device), target_kps.to(device), masks.to(device), masks.to(device)
