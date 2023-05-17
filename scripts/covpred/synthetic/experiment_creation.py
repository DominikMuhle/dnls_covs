import copy
from enum import Enum
from typing import List, Optional, Tuple
from math import sqrt

import torch
import theseus as th

from covpred.common import points_in_img, to_3d_point
from covpred.config.synthetic.config import SyntheticConfig
from covpred.dataset.synthetic_dataset import SyntheticFramePairs, SyntheticFrame
from covpred.math.common import rotation_matrix_2d

PNECProblems = Tuple[torch.Tensor, torch.Tensor]


img_size = (350, 1200)
focal_length = 700.0
K = torch.diag(torch.tensor([focal_length, focal_length, 1.0]))
K[0, 2] = img_size[1] / 2
K[1, 2] = img_size[0] / 2
K_inv = torch.linalg.inv(K)


def point_generation(num_points: int, mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    return torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance).rsample(
        (1.0, num_points)
    )


def gen_world_points_in_focal_region(num_points: int, host_pose: th.SE3) -> torch.Tensor:
    mult = torch.tensor([[img_size[1], img_size[0]]])

    host_img_points = torch.rand(num_points, 2) * mult
    host_depths = torch.rand(num_points) * 5 + 1
    world_points = img_to_world(host_img_points, host_depths, host_pose)
    return world_points


def generate_point_in_focal_region(num_points: int, host_pose: th.SE3, target_pose: th.SE3) -> torch.Tensor:
    max_trys = 10 * num_points
    mult = torch.tensor([[img_size[1], img_size[0]]])

    host_img_points = torch.rand(max_trys, 2) * mult
    host_depths = torch.rand(max_trys) * 5 + 1
    world_points = img_to_world(host_img_points, host_depths, host_pose)
    target_img_points, target_depths = world_to_img(world_points, target_pose)
    target_img_points = target_img_points[0]

    in_bounds = points_in_img(target_img_points, img_size, (5, 5))

    return world_points[in_bounds][:num_points, ...]


def relative_pose_generation(scale_rotation: float = 1.0, scale_translation: float = 1.0):
    max_angle = scale_rotation * torch.pi / 180
    max_translation = scale_translation
    angle_axis = torch.rand(3) / sqrt(3) * max_angle
    translation = torch.rand(3) / sqrt(3) * max_translation
    se3 = torch.concat([translation, angle_axis], 0)[None, ...]
    return th.SE3().exp_map(se3)


def cov_from_parameters(scale: float, alpha: float, beta: float) -> torch.Tensor:
    rot = rotation_matrix_2d(alpha)
    return scale * torch.einsum("ij,jk,lk->il", rot, torch.diag(torch.tensor([beta, 1 - beta])), rot)


def uncertainty_generation(
    num_points: int, max_alpha=torch.pi, max_beta=0.9, max_scale=1.0, min_scale=0.1
) -> torch.Tensor:
    covariances = []
    for _ in range(num_points):
        alpha = torch.rand(1, 1) * max_alpha
        beta = (torch.rand(1, 1) * (max_beta - 0.5)) + 0.5
        scale = 2.0 * (torch.rand(1, 1) * (max_scale - min_scale) + min_scale)
        covariances.append(cov_from_parameters(scale[0, 0].item(), alpha[0, 0].item(), beta[0, 0].item()))
    return torch.stack(covariances)


def uniform_uncertainty_generation(num_points: int, scale: float = 0.5) -> torch.Tensor:
    return scale * torch.eye(2)[None].expand(num_points, -1, -1)


def init_covariances(
    num_points: int,
    random: bool = True,
    offset=True,
    gt_covariances: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if random:
        if offset and gt_covariances is not None:
            offset_cov = uncertainty_generation(num_points, 0.04 * torch.pi, 0.60, 1.1, 0.9)
            return torch.einsum("...ij,...jk,...lk->...il", offset_cov, gt_covariances, offset_cov)
        else:
            return uncertainty_generation(num_points)
    else:
        return 0.5 * torch.stack(num_points * [torch.eye(2)])


def sample_noisy_points(points_2d: torch.Tensor, covariances: torch.Tensor) -> torch.Tensor:
    mean = torch.zeros(2)
    noisy_points = []
    for point_2d, covariance in zip(points_2d, covariances):
        noise = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance).rsample()
        noisy_points.append(point_2d + noise)
    return torch.stack(noisy_points)


def sample_noisy_batched_points(points_2d: torch.Tensor, covariances: torch.Tensor) -> torch.Tensor:
    # points_2d # B,N,2
    # covariances # N,2,2
    noisy_points = []
    mean = torch.zeros_like(covariances)[..., 0]  # N,2

    for idx, (mu, sigma) in enumerate(zip(mean, covariances)):
        noise = torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma).rsample((points_2d.shape[0],))
        noisy_points.append(points_2d[:, idx, :] + noise)
    return torch.stack(noisy_points, dim=1)


def unproject(img_points: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    intermediate_points = torch.einsum("ij,...j->...i", K_inv, to_3d_point(img_points))
    return intermediate_points * (depths / intermediate_points[..., -1])[..., None]


def project(points: torch.Tensor) -> torch.Tensor:
    intermediate_points = points / points[..., -1][..., None]
    image_points = (torch.einsum("ij,...j->...i", K, intermediate_points))[..., :2]
    return image_points


def img_to_world(img_points: torch.Tensor, depths: torch.Tensor, camera_pose: th.SE3) -> torch.Tensor:
    return camera_pose.transform_from(unproject(img_points, depths)).tensor


def world_to_img(points: torch.Tensor, camera_pose: th.SE3) -> Tuple[torch.Tensor, torch.Tensor]:
    # über die Punkte iterieren
    camera_points = torch.stack(
        [camera_pose.transform_to(single_points[None]).tensor for single_points in points], dim=1
    )
    # camera_points = camera_pose.transform_to(points).tensor
    return project(camera_points), camera_points[..., 2]


def project_points_to_image(pose: th.SE3, points: torch.Tensor) -> torch.Tensor:
    camera_points = (
        torch.matmul(pose.inverse().tensor[0, :, :3], points[0, ...].transpose(0, 1)).transpose(0, 1)
        + pose.inverse().tensor[0, :, 3][None, ...]
    )
    camera_points = camera_points / camera_points[..., -1][..., None]
    image_points = (torch.einsum("ij,...j->...i", K, camera_points))[..., :2]
    return image_points


def generate_single_pose(poses_0: th.SE3, max_t: float, max_r: float) -> th.SE3:
    num_poses = poses_0.tensor.shape[0]
    angle_axis = torch.rand(3) / sqrt(3) * max_r * torch.pi / 180.0
    translation = torch.rand(3) / sqrt(3) * max_t
    se3 = torch.concat([translation, angle_axis], 0)[None, ...].expand(num_poses, -1)
    return th.SE3.exp_map(se3).compose(poses_0)


def generate_individual_poses(poses_0: th.SE3, max_t: float, max_r: float) -> th.SE3:
    num_poses = poses_0.tensor.shape[0]
    translations = torch.rand(num_poses, 3) / sqrt(3) * max_t
    # translation_length = torch.linalg.norm(translations, dim=-1)

    z_dir = torch.zeros_like(translations)
    z_dir[..., 2] = 1.0
    normal_vector = torch.cross(translations, z_dir)
    rotation_vector = normal_vector / torch.linalg.norm(normal_vector, dim=-1, keepdim=True)  # B,3
    angle = torch.randn(translations.shape[0]) * max_r * torch.pi / 180.0  # B

    quaternions = torch.cat(
        [
            translations.permute(1, 0),
            torch.cos(angle)[None],
            (rotation_vector * torch.sin(angle)[:, None]).permute(1, 0),
        ]
    )  # 7,B

    return th.SE3(x_y_z_quaternion=quaternions.permute(1, 0)).compose(poses_0)


class TrainingFrames(Enum):
    Host = 1
    Target = 2
    Both = 3


def create_problems(
    config: SyntheticConfig,
) -> Tuple[SyntheticFramePairs, th.SE3]:
    training_frames = TrainingFrames[config.training_frames]
    random_cov_init = False
    poses_0 = th.SE3(tensor=torch.eye(4)[None, :3, :])
    world_points = gen_world_points_in_focal_region(config.num_points, poses_0)
    poses_0 = th.SE3(tensor=poses_0.tensor.expand(config.num_problems, -1, -1))

    if config.individual_poses:
        poses_1 = generate_individual_poses(poses_0, config.max_t, config.max_r)
    else:
        poses_1 = generate_single_pose(poses_0, config.max_t, config.max_r)

    image_points_0, depth_0 = world_to_img(world_points, poses_0)
    image_points_1, depth_1 = world_to_img(world_points, poses_1)

    if training_frames in [TrainingFrames.Host, TrainingFrames.Both]:
        covs_0 = uncertainty_generation(config.num_points, max_scale=1.0)
        covs_0_init = init_covariances(config.num_points, random_cov_init, True, covs_0)
    else:
        covs_0 = uniform_uncertainty_generation(config.num_points, scale=0.01)
        covs_0_init = copy.deepcopy(covs_0)

    if training_frames in [TrainingFrames.Target, TrainingFrames.Both]:
        covs_1 = uncertainty_generation(config.num_points, max_scale=1.0)
        covs_1_init = init_covariances(config.num_points, random_cov_init, True, covs_1)
    else:
        covs_1 = uniform_uncertainty_generation(config.num_points, scale=0.01)
        covs_1_init = copy.deepcopy(covs_1)

    img_pts_0_noisy = sample_noisy_batched_points(image_points_0, covs_0)
    img_pts_1_noisy = sample_noisy_batched_points(image_points_1, covs_1)

    return (
        (
            SyntheticFramePairs(
                SyntheticFrame(image_points_0, img_pts_0_noisy, covs_0, covs_0_init),
                SyntheticFrame(image_points_1, img_pts_1_noisy, covs_1, covs_1_init),
            )
        ),
        poses_0.between(poses_1),
    )


def create_noisy_problems(
    config: SyntheticConfig,
) -> Tuple[SyntheticFramePairs, th.SE3]:
    training_frames = TrainingFrames[config.training_frames]
    random_cov_init = False
    poses_0 = th.SE3(tensor=torch.eye(4)[None])
    world_points = gen_world_points_in_focal_region(config.num_points, poses_0)
    poses_0 = th.SE3(tensor=poses_0.tensor.expand(config.num_problems, -1, -1))

    if config.individual_poses:
        poses_1 = generate_individual_poses(poses_0, config.max_t, config.max_r)
    else:
        poses_1 = generate_single_pose(poses_0, config.max_t, config.max_r)

    image_points_0, depth_0 = world_to_img(world_points, poses_0)
    image_points_1, depth_1 = world_to_img(world_points, poses_1)

    if training_frames in [TrainingFrames.Host, TrainingFrames.Both]:
        covs_0 = uncertainty_generation(config.num_points, max_scale=1.0)
        covs_0[0] = covs_0[0] * 10
        covs_0_init = init_covariances(config.num_points, random_cov_init, True, covs_0)
    else:
        covs_0 = uniform_uncertainty_generation(config.num_points, scale=0.01)
        covs_0_init = copy.deepcopy(covs_0)

    if training_frames in [TrainingFrames.Target, TrainingFrames.Both]:
        covs_1 = uncertainty_generation(config.num_points, max_scale=1.0)
        covs_1[0] = covs_1[0] * 10
        covs_1_init = init_covariances(config.num_points, random_cov_init, True, covs_1)
    else:
        covs_1 = uniform_uncertainty_generation(config.num_points, scale=0.01)
        covs_1_init = copy.deepcopy(covs_1)

    img_pts_0_noisy = sample_noisy_batched_points(image_points_0, covs_0)
    img_pts_1_noisy = sample_noisy_batched_points(image_points_1, covs_1)

    return (
        (
            SyntheticFramePairs(
                SyntheticFrame(image_points_0, img_pts_0_noisy, covs_0, covs_0_init),
                SyntheticFrame(image_points_1, img_pts_1_noisy, covs_1, covs_1_init),
            )
        ),
        poses_0.between(poses_1),
    )


def create_outlier_problems(
    config: SyntheticConfig,
) -> Tuple[SyntheticFramePairs, th.SE3]:
    training_frames = TrainingFrames[config.training_frames]
    random_cov_init = False
    poses_0 = th.SE3(tensor=torch.eye(4)[None])
    world_points = gen_world_points_in_focal_region(config.num_points, poses_0)
    poses_0 = th.SE3(tensor=poses_0.tensor.expand(config.num_problems, -1, -1))

    if config.individual_poses:
        poses_1 = generate_individual_poses(poses_0, config.max_t, config.max_r)
    else:
        poses_1 = generate_single_pose(poses_0, config.max_t, config.max_r)

    image_points_0, depth_0 = world_to_img(world_points, poses_0)
    image_points_0[:, 0] = image_points_0[:, 0] + torch.tensor([1.0, 1.0])
    image_points_1, depth_1 = world_to_img(world_points, poses_1)

    if config.training_frames in [TrainingFrames.Host, TrainingFrames.Both]:
        covs_0 = uncertainty_generation(config.num_points, max_scale=1.0)
        covs_0_init = init_covariances(config.num_points, random_cov_init, True, covs_0)
    else:
        covs_0 = uniform_uncertainty_generation(config.num_points, scale=0.01)
        covs_0_init = copy.deepcopy(covs_0)

    if config.training_frames in [TrainingFrames.Target, TrainingFrames.Both]:
        covs_1 = uncertainty_generation(config.num_points, max_scale=1.0)
        covs_1_init = init_covariances(config.num_points, random_cov_init, True, covs_1)
    else:
        covs_1 = uniform_uncertainty_generation(config.num_points, scale=0.01)
        covs_1_init = copy.deepcopy(covs_1)

    img_pts_0_noisy = sample_noisy_batched_points(image_points_0, covs_0)
    img_pts_1_noisy = sample_noisy_batched_points(image_points_1, covs_1)

    return (
        (
            SyntheticFramePairs(
                SyntheticFrame(image_points_0, img_pts_0_noisy, covs_0, covs_0_init),
                SyntheticFrame(image_points_1, img_pts_1_noisy, covs_1, covs_1_init),
            )
        ),
        poses_0.between(poses_1),
    )
