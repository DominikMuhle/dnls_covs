import torch


def add_jitter(keypoints: torch.Tensor, std: float) -> torch.Tensor:
    assert keypoints.ndim == 2
    keypoints[keypoints[:, 1] < 150] += std * torch.randn_like(keypoints[keypoints[:, 1] < 150])
    keypoints[:, 0].clamp_(min=0.0, max=1225.0)
    keypoints[:, 1].clamp_(min=0.0, max=369.0)
    return keypoints
