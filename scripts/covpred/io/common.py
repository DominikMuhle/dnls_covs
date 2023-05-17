from pathlib import Path

import torch
import numpy as np


def load_camera_calibration_kitti(path: Path) -> torch.Tensor:
    calib_path = path.joinpath("calib.txt")
    if path.stem == "00":
        calib = np.genfromtxt(calib_path, skip_header=1, max_rows=3)
    else:
        calib = np.genfromtxt(calib_path, max_rows=1)[1:].reshape(3, 4)
    return torch.from_numpy(calib[:3, :3])


def load_camera_calibration_kitti_raw(path: Path) -> torch.Tensor:
    calib_path = path.joinpath("calib.txt")
    cam_calib_file_data = {}
    with open(calib_path, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            try:
                cam_calib_file_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
            except ValueError:
                pass
    return torch.from_numpy(np.reshape(cam_calib_file_data["P_rect_00"], (3, 4)))


def load_camera_calibration_euroc(path: Path) -> torch.Tensor:
    calib_path = path.joinpath("camera.txt")
    print(f"Loading camera calibration from {str(calib_path)}")
    parameters = np.genfromtxt(calib_path, max_rows=1)[1:-1]
    np_K = np.eye(3)
    np_K[0, 0] = parameters[0]
    np_K[1, 1] = parameters[1]
    np_K[0, 2] = parameters[2]
    np_K[1, 2] = parameters[3]
    return torch.from_numpy(np_K)


def load_poses(path: Path, filename: str = "poses.txt") -> torch.Tensor:
    poses_path = path.joinpath(filename)
    poses = np.genfromtxt(poses_path).reshape(-1, 3, 4)
    return torch.from_numpy(poses)


def save_poses(path: Path, poses: torch.Tensor):
    np.savetxt(path, poses.flatten(-2, -1).cpu().detach().numpy())
