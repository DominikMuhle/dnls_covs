from typing import List, Tuple
import numpy as np
import torch
import pypnec


def KLT(
    images: torch.Tensor,
    K_inv: torch.Tensor,
    host_idx: torch.Tensor,
    target_idx: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # images: B,N,1,H,W
    # K_inv: B,N,3,3
    K_matrices = torch.linalg.inv(K_inv)
    host_kps = []
    target_kps = []
    masks = []
    for i in range(host_idx.shape[0]):  # batch entries
        for j in range(host_idx.shape[1]):  # pairs
            host_K = K_matrices[i, host_idx[i, j]]
            host_img = (images[i, host_idx[i, j]][0] * 255.0).to("cpu").detach().numpy()
            target_K = K_matrices[i, target_idx[i, j]]
            target_img = (images[i, target_idx[i, j]][0] * 255.0).to("cpu").detach().numpy()
            # tracking pos p' in the host frame and the same pos in the target frame p'' are given by p''=p'+offset
            offset = (target_K - host_K)[:2, 2].to("cpu").detach().numpy()

            host_tracks = pypnec.Keypoints()
            target_tracks = pypnec.Keypoints()

            pypnec.KLTImageMatching(host_img, target_img, host_tracks, target_tracks, offset, False)
            host_kps.append(torch.from_numpy(np.array(list(host_tracks))))
            target_kps.append(torch.from_numpy(np.array(list(target_tracks))))
            masks.append(torch.ones_like(host_kps[-1])[..., 0])

    host_kps = torch.nn.utils.rnn.pad_sequence(host_kps, True, 0.0)
    target_kps = torch.nn.utils.rnn.pad_sequence(target_kps, True, 0.0)
    masks = torch.nn.utils.rnn.pad_sequence(masks, True, 0.0)
    flip = False
    if flip:
        host_kps = host_kps.flip(-1)
        target_kps = target_kps.flip(-1)

    return host_kps.to(device), target_kps.to(device), masks.to(device), masks.to(device)
