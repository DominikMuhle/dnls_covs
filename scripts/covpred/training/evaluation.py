from typing import Dict

import torch
from ignite.engine import Engine
import ignite.distributed as idist
from ignite.metrics import Metric
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
import theseus as th

from covpred.loss_functions.metrics import RotationalError, FilteredRotationalError, CycleError, RotationalRMSE
from covpred.common import (
    Initializations,
    filter_keypoints,
    get_initialization,
    get_rel_poses,
    create_bearing_vectors,
    get_entries_in_batch,
    subsample_points,
)
from covpred.config.training import Config
from covpred.model.optimization.theseus.optimization_layer import TheseusLayer
from covpred.model.full_model import DeepPNEC, extract_covs, extract_covs_alt
from covpred.model.transforms import covs_to_image
from covpred.training.common import best_init_pose
from covpred.visualization.dense import dense_covariance_images
from covpred.visualization.keypoints import keypoints_in_img
from covpred.visualization.common import VisualizationImage


# TODO: Init evaluation for NEC, NEC_LS, KLT_NEC
def create_evaluator(
    model: DeepPNEC,
    theseus_opt: TheseusLayer,
    config: Config,
) -> Engine:
    device = idist.device()

    @torch.no_grad()
    def evaluate_step(engine: Engine, batch):
        model.eval()
        batch_size = batch[0].shape[0]

        # permute is important due to collate
        images = batch[0].to(device)  # B,N,1,H,W; torch.float32
        K_inv = batch[1].to(device)  # B,N,3,3; torch.float64
        gt_poses = th.SE3(tensor=batch[2].to(device).flatten(0, 1))  # B,N,3,3; torch.float64
        host_keypoints = batch[3].permute(0, 2, 1, 3).to(device)  # B,P,M,2
        host_img_idx = batch[4]  # B,P; torch.int32
        target_keypoints = batch[5].permute(0, 2, 1, 3).to(device)  # B,P,M,2
        target_img_idx = batch[6]  # B,P; torch.int32
        masks = batch[7].to(device).permute(0, 2, 1)  # B,P,M; torch.int32

        filter_keypoints(host_keypoints, target_keypoints, masks, (images.shape[-2], images.shape[-1]))

        host_K_inv = get_entries_in_batch(K_inv, host_img_idx, batch_size)
        target_K_inv = get_entries_in_batch(K_inv, target_img_idx, batch_size)

        if config.dataset.sweep.self_supervised:
            init_poses = get_initialization(Initializations.IDENTITY, num_poses=gt_poses.shape[0])
            init_poses.to(device)
            init_poses.to(host_keypoints.dtype)
        else:
            init_poses = th.SE3(tensor=batch[9].flatten(0, 1).to(device))  # B,P,3,4
        nec_poses = th.SE3(tensor=batch[10].flatten(0, 1).to(device))  # B,P,3,4

        # prediction_start.record()
        cov_predictions = model(images)
        host_covs, target_covs = extract_covs(
            cov_predictions, host_keypoints, host_img_idx, target_keypoints, target_img_idx
        )
        host_bvs, host_bvs_covs = create_bearing_vectors(
            host_keypoints.flatten(0, 1),
            host_covs.flatten(0, 1),
            host_K_inv,
        )
        target_bvs, target_bvs_covs = create_bearing_vectors(
            target_keypoints.flatten(0, 1), target_covs.flatten(0, 1), target_K_inv, masks.flatten(0, 1)
        )

        if config.dataset.sweep.self_supervised:
            init_poses.tensor[..., :3, 3] = nec_poses[..., :3, 3]
        pose_choice = [init_poses]

        # deep_init = best_init_pose(
        #     pose_choice,
        #     host_bvs,
        #     host_bvs_covs,
        #     target_bvs,
        #     target_bvs_covs,
        #     config.pose_estimation,
        # )
        deep_init = init_poses
        # th_poses, info = theseus_opt(
        #     host_bvs,
        #     host_bvs_covs,
        #     target_bvs,
        #     target_bvs_covs,
        #     deep_init,
        #     config.theseus,
        #     config.pose_estimation,
        #     device,
        #     torch.float64,
        #     False,
        # )
        th_poses, info = theseus_opt(
            host_bvs,
            host_bvs_covs,
            target_bvs,
            target_bvs_covs,
            deep_init,
        )

        return th_poses.tensor.detach(), gt_poses.tensor.detach()

    evaluator = Engine(evaluate_step)

    val_metrics: Dict[str, Metric] = {
        "rotational error": RotationalError(device=device),
        "filtered rotational error": FilteredRotationalError(config.hyperparameter.error_threshold, device=device),
        "rotational RMSE": RotationalRMSE(device=device),
    }
    if config.dataset.sweep.self_supervised:
        val_metrics["cycle error"] = CycleError(config.dataset.sweep.tuple_length, device=device)

    for name, metric in val_metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_visualization_evaluator(
    model: DeepPNEC, config: Config, tb_logger: TensorboardLogger, training_engine: Engine
) -> Engine:
    device = idist.device()

    @torch.no_grad()
    def visualization_step(engine: Engine, batch):
        # expects batch size to be 1!!!
        model.eval()
        batch_size = batch[0].shape[0]
        tuple_length = batch[0].shape[1]

        # permute is important due to collate
        images = batch[0][0][None].to(device)  # B,N,1,H,W; torch.float32
        host_keypoints = batch[3][0][None].permute(0, 2, 1, 3).to(device)  # B,P,M,2
        host_img_idx = batch[4][0][None]  # B,P; torch.int32
        target_keypoints = batch[5][0][None].permute(0, 2, 1, 3).to(device)  # B,P,M,2
        target_img_idx = batch[6][0][None]  # B,P; torch.int32
        masks = batch[7].to(device).permute(0, 2, 1)  # B,P,M; torch.int32

        filter_keypoints(host_keypoints, target_keypoints, masks, (images.shape[-2], images.shape[-1]))

        cov_predictions = model(images)  # 1*N,H,W,2,2
        host_covs, target_covs = extract_covs(
            cov_predictions,
            host_keypoints,
            host_img_idx,
            target_keypoints,
            target_img_idx,
        )  # 1*P,M,2,2, 1*P,M,2,2

        dense_images = [
            VisualizationImage(img) for img in torch.from_numpy(covs_to_image(cov_predictions, (None, 5.0)))
        ]
        # TODO: maybe deal with tuple of length > 2
        subsampling = 10
        kp_images = [
            VisualizationImage(
                images[0, 0].permute(1, 2, 0),
                None,
                subsample_points(host_keypoints[0, 0], subsampling),
                subsample_points(host_covs[0, 0], subsampling),
            ),
            VisualizationImage(
                images[0, 1].permute(1, 2, 0),
                None,
                subsample_points(target_keypoints[0, 0], subsampling),
                subsample_points(target_covs[0, 0], subsampling),
            ),
        ]

        # TODO: use the correct scaling for dense cov images. Maybe add explanation plot for hsv
        fig, axs = dense_covariance_images(dense_images)
        tb_logger.writer.add_figure(
            f"Dense Covariances/{engine.state.iteration}",
            figure=fig,
            global_step=training_engine.state.epoch,
            close=True,
        )

        fig, axs = keypoints_in_img(kp_images, color="r", kp_size=1.0, scale=6.0, linewidth=2.0)
        tb_logger.writer.add_figure(
            f"Keypoints with Covariances/{engine.state.iteration}",
            figure=fig,
            global_step=training_engine.state.epoch,
            close=True,
        )

    evaluator = Engine(visualization_step)

    return evaluator
