import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union, overload
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import theseus as th
import ignite.distributed as idist

from covpred.common import Initializations, create_bearing_vectors, get_initialization, DummySummaryWriter
from covpred.config.pose_estimation.config import NECConfig, PoseEstimationConfig
from covpred.config.synthetic_config import SyntheticExperimentConfig
from covpred.dataset.synthetic_dataset import SyntheticDataset
from covpred.evaluation.metrics.covariance_similarity import covariance_similarity
from covpred.model.optimization.theseus.optimization_layer import TheseusLayer, create_theseus_layer
from covpred.model.output_filter import OutputFilter
from covpred.model.parametrization import BackTransform, BaseParametrization
from covpred.config.theseus.config import TheseusConfig
from covpred.math.common import angular_diff
from covpred.synthetic.experiment_creation import K, K_inv, TrainingFrames
from covpred.model.optimization.eigenvalue import nec
from covpred.visualization.gradients import gradient_vis
from covpred.visualization.keypoints import keypoints_in_img, covariance_grid
from covpred.visualization.common import img_frame, VisualizationImage


def save_covariances(path: Path, covariances: Dict[str, torch.Tensor]) -> None:
    path.mkdir(exist_ok=True, parents=True)
    for name, covs in covariances.items():
        np.savetxt(path.joinpath(name + ".txt"), covs.detach().cpu().flatten(-2, -1).numpy())


@overload
def train_epoch(
    host_covs_param: torch.nn.Parameter,
    target_covs_param: torch.nn.Parameter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.modules.loss._Loss,
    theseus_opt: TheseusLayer,
    back_transform_fn: BackTransform,
    output_filters: OutputFilter,
    gradients: List[torch.Tensor],
    device: str,
    pose_estimation_config: PoseEstimationConfig,
    init: Literal[False] = False,
) -> torch.Tensor:
    ...


@overload
def train_epoch(
    host_covs_param: torch.nn.Parameter,
    target_covs_param: torch.nn.Parameter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.modules.loss._Loss,
    theseus_opt: TheseusLayer,
    back_transform_fn: BackTransform,
    output_filters: OutputFilter,
    gradients: List[torch.Tensor],
    device: str,
    pose_estimation_config: PoseEstimationConfig,
    init: Literal[True],
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


# @overload
# def train_epoch(
#     host_covs_param: torch.nn.Parameter,
#     target_covs_param: torch.nn.Parameter,
#     dataloader: DataLoader,
#     optimizer: torch.optim.Optimizer,
#     loss_fn: torch.nn.modules.loss._Loss,
#     theseus_config: TheseusConfig,
#     back_transform_fn: BackTransform,
#     output_filters: OutputFilter,
#     gradients: List[torch.Tensor],
#     device: str,
#     init: bool = False,
# ) -> torch.Tensor:
#     ...


def train_epoch(
    host_covs_param: torch.nn.Parameter,
    target_covs_param: torch.nn.Parameter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.modules.loss._Loss,
    theseus_opt: TheseusLayer,
    back_transform_fn: BackTransform,
    output_filters: OutputFilter,
    gradients: List[torch.Tensor],
    device: str,
    pose_estimation_config: PoseEstimationConfig,
    init: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    angular_errors = []
    nec_errors = []
    gt_covs_errors = []
    for batch_idx, data in enumerate(dataloader):
        host_points = data[0].to(device)
        target_points = data[1].to(device)
        optimizer.zero_grad()
        gt_poses = th.SE3(tensor=data[2].to(device).to(torch.float64))
        num_problems = host_points.shape[0]

        host_covs = back_transform_fn(output_filters.filter_output(host_covs_param), False, False)
        target_covs = back_transform_fn(output_filters.filter_output(target_covs_param), False, False)
        target_covs.retain_grad()

        host_bvs, host_bvs_covs = create_bearing_vectors(host_points, host_covs, K_inv.expand(num_problems, -1, -1))
        target_bvs, target_bvs_covs = create_bearing_vectors(
            target_points, target_covs, K_inv.expand(num_problems, -1, -1)
        )

        init_poses = get_initialization(
            Initializations.RANDOM_PERTUBATION, gt_poses, max_angle=0.05, max_translation=0.05
        )

        if init:
            nec_poses = []
            for bvs_0, bvs_1, init_pose in zip(host_bvs, target_bvs, init_poses.tensor):
                nec_poses.append(
                    nec(bvs_0, bvs_1, th.SE3(tensor=init_pose[None]), pose_estimation_config.nec)[0].tensor
                )
                # nec_poses.append(nec(bvs_0, bvs_1, th.SE3(tensor=init_pose[None]), ransac=False)[0].tensor)
            nec_poses = th.SE3(tensor=torch.concat(nec_poses))
            nec_poses.to(device)

            nec_err = angular_diff(
                gt_poses,
                nec_poses,
                degree=True,
            )
            nec_errors.append(nec_err.detach().to("cpu"))

            host_bvs_alt, host_bvs_covs_alt = create_bearing_vectors(
                host_points, dataloader.dataset.image_pairs.host.covs, K_inv.expand(num_problems, -1, -1)
            )
            target_bvs_alt, target_bvs_covs_alt = create_bearing_vectors(
                target_points, dataloader.dataset.image_pairs.target.covs, K_inv.expand(num_problems, -1, -1)
            )

            gt_covs_poses, gt_info = theseus_opt(
                host_bvs_alt, host_bvs_covs_alt, target_bvs_alt, target_bvs_covs_alt, init_poses
            )
            gt_covs_error = angular_diff(gt_poses, gt_covs_poses, degree=True)
            gt_covs_errors.append(gt_covs_error.detach().to("cpu"))

        th_poses, info = theseus_opt(host_bvs, host_bvs_covs, target_bvs, target_bvs_covs, init_poses)

        # th_poses, info = theseus_opt(
        #     host_bvs,
        #     host_bvs_covs,
        #     target_bvs,
        #     target_bvs_covs,
        #     init_poses,
        #     theseus_config,
        #     device,
        #     torch.float64,
        #     False,
        # )

        errors = angular_diff(
            gt_poses,
            th_poses,
            degree=True,
        )
        filtered_errors = errors[errors < 0.1]
        angular_errors.append(errors.detach().to("cpu"))
        final_loss = loss_fn(filtered_errors, torch.zeros_like(filtered_errors))
        final_loss.backward()

        if not init:
            gradients.append(target_covs.grad.detach().to("cpu"))

            optimizer.step()

    if not init:
        return torch.concat(angular_errors, dim=0)
    return torch.concat(angular_errors, dim=0), torch.concat(nec_errors, dim=0)


def train(
    dataset: SyntheticDataset,
    config: SyntheticExperimentConfig,
    parametrization: BaseParametrization,
    output_filter: OutputFilter,
    device: str,
    training_frames: TrainingFrames,
    path: Path,
    writer: Union[SummaryWriter, DummySummaryWriter],
    num_epochs: int = 100,
    batch_size: int = 640,
) -> None:
    logger = logging.getLogger("synthetic")

    precision = torch.float64
    theseus_opt = create_theseus_layer(
        config.theseus, config.pose_estimation, idist.device(), config.synthetic.num_points, precision, True, False
    )

    save_covariances(
        path,
        {
            "covariances_host_gt": dataset.image_pairs.host.covs,
            "covariances_target_gt": dataset.image_pairs.target.covs,
        },
    )
    save_covariances(
        path,
        {
            f"covariances_host_0": dataset.image_pairs.host.init_covs,
            f"covariances_target_0": dataset.image_pairs.target.init_covs,
        },
    )

    parametrized_host_covs = output_filter.inv_filter_output(
        parametrization.transform()(dataset.image_pairs.host.init_covs.to(device))
    )
    host_param = torch.nn.Parameter(parametrized_host_covs)
    parametrized_target_covs = output_filter.inv_filter_output(
        parametrization.transform()(dataset.image_pairs.target.init_covs.to(device))
    )
    target_param = torch.nn.Parameter(parametrized_target_covs)
    param_list: List[torch.nn.parameter.Parameter] = []

    if training_frames in [TrainingFrames.Host, TrainingFrames.Both]:
        param_list.append(host_param)
    if training_frames in [TrainingFrames.Target, TrainingFrames.Both]:
        param_list.append(target_param)
    cov_optimizer = torch.optim.Adam(param_list, lr=2.0e-3, betas=(0.9, 0.99))

    loss_fn = torch.nn.L1Loss(reduction="mean")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    gradients: List[torch.Tensor] = []
    angular_errors, nec_errors = train_epoch(
        host_param,
        target_param,
        dataloader,
        cov_optimizer,
        loss_fn,
        theseus_opt,
        parametrization.back_transform(),
        output_filter,
        gradients,
        device,
        config.pose_estimation,
        True,
    )
    logger.info(f"NEC error {nec_errors.mean()}")
    logger.info(f"Epoch 0: {angular_errors.mean()}")
    writer.add_scalar(f"rotational error", nec_errors.mean(), global_step=-1)
    writer.add_scalar(f"rotational error", angular_errors.mean(), global_step=0)

    imgs = [
        VisualizationImage(
            img_frame.permute(1, 2, 0), None, dataset.image_pairs.host.pts[0], dataset.image_pairs.host.init_covs
        ),
        VisualizationImage(
            img_frame.permute(1, 2, 0), None, dataset.image_pairs.target.pts[0], dataset.image_pairs.target.init_covs
        ),
    ]
    fig, axs = keypoints_in_img(imgs, color="blue", kp_size=1.0, scale=3.0, linewidth=1.0)

    for name, frame in zip(["host", "target"], [dataset.image_pairs.host, dataset.image_pairs.target]):
        fig, axs = covariance_grid(
            frame.init_covs[:10].reshape(2, 5, 2, 2),
            frame.covs[:10].reshape(2, 5, 2, 2),
            resize_covariances=True,
        )
        writer.add_figure(f"covariance_grid/{name}", fig, global_step=0)
    # TODO: save images to folder

    sigma_errors: List[float] = []
    host_kl_divergence_errors: List[float] = []
    target_kl_divergence_errors: List[float] = []
    errors = covariance_similarity(
        dataset.image_pairs.host.pts,
        dataset.image_pairs.target.pts,
        [dataset.image_pairs.host.init_covs, dataset.image_pairs.host.covs],
        [dataset.image_pairs.target.init_covs, dataset.image_pairs.target.covs],
        K_inv,
        dataset.gt_poses,
        0.0,
    )
    sigma_errors.append(torch.abs(errors["sigma_2"][0, 1]).mean().item())
    host_kl_divergence_errors.append(errors["host_kl_divergence"][0, 1].mean().item())
    target_kl_divergence_errors.append(errors["target_kl_divergence"][0, 1].mean().item())

    for epoch in range(num_epochs):
        angular_errors = train_epoch(
            host_param,
            target_param,
            dataloader,
            cov_optimizer,
            loss_fn,
            theseus_opt,
            parametrization.back_transform(),
            output_filter,
            gradients,
            device,
            config.pose_estimation,
        )
        logger.info(f"Epoch {epoch + 1}: {angular_errors.mean()}")

        writer.add_scalar(f"rotational error", angular_errors.mean(), global_step=epoch + 1)

        est_host_covs = parametrization.back_transform()(output_filter.filter_output(host_param), True, False)
        est_target_covs = parametrization.back_transform()(output_filter.filter_output(target_param), True, False)
        errors = covariance_similarity(
            dataset.image_pairs.host.pts.to(device),
            dataset.image_pairs.target.pts.to(device),
            [est_host_covs.to(device), dataset.image_pairs.host.covs.to(device)],
            [est_target_covs.to(device), dataset.image_pairs.target.covs.to(device)],
            K_inv.to(device),
            dataset.gt_poses,
            0.0,
        )
        sigma_errors.append(torch.abs(errors["sigma_2"][0, 1]).mean().item())
        host_kl_divergence_errors.append(errors["host_kl_divergence"][0, 1].mean().item())
        target_kl_divergence_errors.append(errors["target_kl_divergence"][0, 1].mean().item())

        if (epoch + 1) % 10 == 0:
            save_covariances(
                path,
                {
                    f"covariances_host_{epoch + 1}": est_host_covs,
                    f"covariances_target_{epoch + 1}": est_target_covs,
                },
            )

            # # TODO: save images in folder
            imgs = [
                VisualizationImage(
                    img_frame.permute(1, 2, 0),
                    None,
                    dataset.image_pairs.host.pts[0],
                    est_host_covs,
                ),
                VisualizationImage(
                    img_frame.permute(1, 2, 0),
                    None,
                    dataset.image_pairs.target.pts[0],
                    est_target_covs,
                ),
            ]
            fig, axs = keypoints_in_img(imgs, color="blue", kp_size=1.0, scale=3.0, linewidth=1.0)
            writer.add_figure(f"keypoints and covariances", fig, global_step=epoch + 1)

            for name, frame, covs in zip(
                ["host", "target"],
                [dataset.image_pairs.host, dataset.image_pairs.target],
                [est_host_covs, est_target_covs],
            ):
                fig, axs = covariance_grid(
                    covs[:10].reshape(2, 5, 2, 2),
                    frame.covs[:10].reshape(2, 5, 2, 2),
                    resize_covariances=True,
                )
                writer.add_figure(f"covariance_grid/{name}", fig, global_step=epoch + 1)

            fig, ax = gradient_vis(gradients, figsize=(1.0, 1.0))
            writer.add_figure(f"gradient directions", fig, global_step=epoch + 1)

    # TODO: Visualize sigma error, kl-divergence after training
