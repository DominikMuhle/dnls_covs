from pathlib import Path
import random
from typing import Dict, Any
from logging import Logger
import logging
from collections import deque

import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from ignite.engine import Engine, Events
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
import theseus as th
from omegaconf import OmegaConf

from covpred.common import (
    Initializations,
    OptInfo,
    filter_keypoints,
    get_initialization,
    create_bearing_vectors,
    get_entries_in_batch,
)
from covpred.config.training import Config
from covpred.dataset.live_matches import LiveMatchesDataset
from covpred.io.model import find_checkpoint_ignite, load_checkpoint_ignite, load_network_filters
from covpred.loss_functions.loss_functions import (
    anchor_error,
    consistency_error,
    self_supervised_loss,
    angular_error,
    supervised_loss,
)
from covpred.loss_functions.metrics import RotationalError, FilteredRotationalError, CycleError, RotationalRMSE
from covpred.model.optimization.eigenvalue import nec

# from covpred.model.optimization.theseus.optimization_layer import theseus_opt
from covpred.model.optimization.theseus.optimization_layer import TheseusLayer, create_theseus_layer
from covpred.training.common import best_init_pose
from covpred.model.full_model import DeepPNEC, extract_covs, extract_covs_alt
import covpred.matching as matching
from covpred.visualization.common import VisualizationImage
from covpred.visualization.keypoints import keypoints_in_img


def visualize_training_examples(
    dataset: LiveMatchesDataset,
    matching_fn: matching.MatchingFunction,
    index: int,
    repeat: int,
    tb_logger: TensorboardLogger,
    logger: Logger,
):
    logger.info(f"Visualizing training examples")
    device = idist.device()

    for i in range(repeat):
        batch = dataset[index]
        images = batch[0][None].to(device)
        K_inv = batch[1][None].to(device)  # B,N,3,3; torch.float64
        host_img_idx = batch[3][None].to(device)  # B,P; torch.int32
        target_img_idx = batch[4][None].to(device)  # B,P; torch.int32

        host_K_inv = get_entries_in_batch(K_inv, host_img_idx, 1)
        target_K_inv = get_entries_in_batch(K_inv, target_img_idx, 1)

        matches = matching_fn(images, K_inv, host_img_idx, target_img_idx, device)

        host_keypoints = matches[0]
        target_keypoints = matches[1]
        # confidence = matches[2]
        masks = matches[3]

        filter_keypoints(host_keypoints, target_keypoints, masks, (images.shape[-2], images.shape[-1]))

        kp_images = [
            VisualizationImage(
                images[0, 0].permute(1, 2, 0),
                torch.linalg.inv(host_K_inv[0].to("cpu")),
                host_keypoints[0],
            ),
            VisualizationImage(
                images[0, 1].permute(1, 2, 0),
                torch.linalg.inv(target_K_inv[0].to("cpu")),
                target_keypoints[0],
            ),
        ]
        fig, axs = keypoints_in_img(kp_images, color="r", kp_size=1.0, scale=6.0, linewidth=2.0)
        tb_logger.writer.add_figure(
            f"Training Example",
            figure=fig,
            global_step=i,
            close=True,
        )


def create_trainer(
    model: DeepPNEC,
    matching_fn: matching.MatchingFunction,
    theseus_opt: TheseusLayer,
    config: Config,
    loss_function: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    output_path: Path,
    logger: Logger,
    tb_logger: TensorboardLogger,
) -> Engine:
    device = idist.device()

    use_amp = config.hyperparameter.use_amp
    scaler = GradScaler(enabled=use_amp)

    # theseus_opt = create_theseus_layer(config.theseus, config.pose_estimation, device, 512, precision, True, False)

    def train_step(engine: Engine, batch):
        logger.debug(f"Learning rate: {lr_scheduler.get_last_lr()}")
        model.train()

        batch_size, tuple_length = batch[3].shape[0], batch[3].shape[1]
        num_pairs = batch_size * tuple_length

        images = batch[0].to(device)  # B,N,1,H,W; torch.float32
        K_inv = batch[1].to(device)  # B,N,3,3; torch.float64
        gt_poses = th.SE3(tensor=batch[2].to(device).flatten(0, 1))  # B,P,3,4; torch.float64
        host_img_idx = batch[3].to(device)  # B,P; torch.int32
        target_img_idx = batch[4].to(device)  # B,P; torch.int32

        host_K_inv = get_entries_in_batch(K_inv, host_img_idx, batch_size)
        target_K_inv = get_entries_in_batch(K_inv, target_img_idx, batch_size)

        init_poses = get_initialization(
            Initializations.RANDOM_PERTUBATION, gt_poses=gt_poses, max_angle=0.05, max_translation=0.05
        )

        matches = matching_fn(images, K_inv, host_img_idx, target_img_idx, device)

        host_keypoints = matches[0]
        target_keypoints = matches[1]
        # confidence = matches[2]
        masks = matches[3]

        filter_keypoints(host_keypoints, target_keypoints, masks, (images.shape[-2], images.shape[-1]))

        with autocast(enabled=use_amp):
            cov_predictions = model(images)
            logger.debug(f"\tModel Output: {cov_predictions.dtype}")
            host_covs, target_covs = extract_covs(
                cov_predictions,
                host_keypoints.reshape(batch_size, tuple_length, -1, 2),
                host_img_idx,
                target_keypoints.reshape(batch_size, tuple_length, -1, 2),
                target_img_idx,
            )
            host_bvs, host_bvs_covs = create_bearing_vectors(
                host_keypoints,
                host_covs.flatten(0, 1),
                host_K_inv,
            )
            target_bvs, target_bvs_covs = create_bearing_vectors(
                target_keypoints, target_covs.flatten(0, 1), target_K_inv, masks
            )
            logger.debug(f"\tBearning Vectors: {host_bvs.dtype}, {target_bvs.dtype}")
            logger.debug(f"\tCovariances: {host_bvs_covs.dtype}, {target_bvs_covs.dtype}")

            nec_poses = []
            for idx, (bvs_0, bvs_1, init, mask) in enumerate(zip(host_bvs, target_bvs, init_poses.tensor, masks)):
                results = nec(bvs_0[mask == 1], bvs_1[mask == 1], th.SE3(tensor=init[None]), config.pose_estimation.nec)
                nec_poses.append(results[0].tensor[0])
                masks[idx][mask == 1] = results[1]

            nec_poses = torch.stack(nec_poses, dim=0)
            logger.debug(f"\tNEC Poses: {nec_poses.dtype}")
            target_bvs = target_bvs * masks[..., None]

            pose_choice = [init_poses]

            deep_init = best_init_pose(
                pose_choice,
                host_bvs,
                host_bvs_covs,
                target_bvs,
                target_bvs_covs,
                config.pose_estimation,
            )
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
            logger.debug(f"\tPNEC Pose: {host_bvs.dtype}, {target_bvs.dtype}")

            output: Dict[str, Any] = {}
            rotational_error = angular_error(th_poses, gt_poses)
            if config.dataset.sweep.self_supervised:
                anchor_err = anchor_error(
                    th_poses, th.SE3(tensor=nec_poses), tuple_length=config.dataset.sweep.tuple_length
                )
                consistency_err = consistency_error(th_poses, tuple_length=config.dataset.sweep.tuple_length)
                full_loss, filtered_loss = self_supervised_loss(
                    consistency_err,
                    loss_function,
                    filter_threshold=config.hyperparameter.error_threshold,
                    additional_errors=[(0.001, anchor_err)],
                )
                output = {
                    "rotational_error": rotational_error.mean().item(),
                    "anchor_error": anchor_err.mean().item(),
                    "consitency_error": consistency_err.mean().item(),
                    "full_loss": full_loss.item(),
                    "filtered_loss": filtered_loss.item(),
                }
            else:
                full_loss, filtered_loss = supervised_loss(
                    rotational_error, loss_function, filter_threshold=config.hyperparameter.error_threshold
                )
                output = {
                    "rotational_error": rotational_error.mean().item(),
                    "full_loss": full_loss.item(),
                    "filtered_loss": filtered_loss.item(),
                }
            trainings_loss = filtered_loss
            logger.debug(f"\tLoss: {trainings_loss.dtype}")

        scaler.scale(trainings_loss).backward()

        # call for iteration == 1 to remove warning
        if engine.state.iteration % config.hyperparameter.accumulation_steps == 0 or engine.state.iteration == 1:
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            # optimizer.step()
            # optimizer.zero_grad(set_to_none=True)

        output["opt_info"] = {
            "last_err": info.last_err.detach().cpu(),
            "converged_iter": info.converged_iter.detach().cpu(),
        }
        if info.err_history is not None:
            info.err_history.nan_to_num_(-1, -1, -1)
            output["opt_info"]["err_history"] = info.err_history.detach().cpu()

        # logger.info(f"mem allocated {torch.cuda.memory_allocated()}")
        # logger.info(f"max mem allocated {torch.cuda.max_memory_allocated()}")
        return output

    trainer = Engine(train_step)
    trainer.logger = logger

    def log_info_training(engine: Engine):
        if "optimizer_info" in engine.state.metrics:
            num_opt, max_iter, not_converged, diverged, bad_gn = engine.state.metrics["optimizer_info"]
            engine.logger.info(f"It took at most {max_iter} iterations to converge.")
            if not_converged > 0:
                engine.logger.warning(f"{not_converged} out of {num_opt} did not converge!")
            if bad_gn > 0:
                engine.logger.warning(f"{bad_gn} out of {num_opt} had a bad last gauss-newton step!")
            if diverged > 0:
                engine.logger.warning(f"{diverged} out of {num_opt} diverged!!!")

    if logger.level == logging.INFO:
        optim_info = OptInfo(
            output_transform=lambda x: {
                "converged_iter": x["opt_info"]["converged_iter"],
                "last_err": x["opt_info"]["last_err"],
                "err_history": x["opt_info"].get("err_history", None),
            }
        )
        optim_info.attach(trainer, "optimizer_info")

        # trainer.add_event_handler(Events.EPOCH_COMPLETED, log_info_training)

    batch_sizes = deque(maxlen=config.logging.tensorboard.running_average_length)
    loss_history: Dict[str, deque] = {}

    # running average logging to tensorboard
    @trainer.on(Events.ITERATION_COMPLETED)
    def update_training_loss(trainer: Engine):
        for name, loss in trainer.state.output.items():
            if not isinstance(loss, float):
                continue
            else:
                if name not in loss_history:
                    loss_history[name] = deque(maxlen=config.logging.tensorboard.running_average_length)
                loss_history[name].append(loss)

        batch_sizes.append(trainer.state.batch[3].shape[0])

    @trainer.on(Events.ITERATION_COMPLETED(every=config.logging.tensorboard.loss_iterations))
    def log_training_loss(trainer: Engine):
        sizes = torch.tensor(batch_sizes)
        tb_losses = {}
        for name, losses in loss_history.items():
            tb_losses[name] = torch.sum(torch.tensor(losses) * sizes) / torch.sum(sizes)

        tb_logger.writer.add_scalars("training_losses", tb_losses, global_step=trainer.state.iteration)

    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}

    common.setup_common_training_handlers(
        trainer=trainer,
        to_save=to_save,
        save_every_iters=100,
        lr_scheduler=lr_scheduler,
        output_path=str(output_path),
        output_names=None,
        with_pbars=False,
        clear_cuda_cache=False,
        log_every_iters=100,
        n_saved=10,
        stop_on_nan=False,
    )

    if config.model.date is not None:
        checkpoint_path = find_checkpoint_ignite(config.model, logger)
        if checkpoint_path is not None:
            logger.info(f"Loading checkpoint from path {checkpoint_path}")
            load_checkpoint_ignite(checkpoint_path, to_save, config.dataset)
            logger.info(f"Loading output filter from {checkpoint_path}")
            model.output_filter = load_network_filters(checkpoint_path, config.model)

    # save config file
    model_cfg_path = output_path.joinpath("model_output.yaml")
    if not model_cfg_path.is_file():
        logger.info(f"Saving model output config to {model_cfg_path}")
        with open(model_cfg_path, "w") as f:
            OmegaConf.save(config.model.output, f)

    data_cfg_path = output_path.joinpath("dataset.yaml")
    if not data_cfg_path.is_file():
        logger.info(f"Saving model output config to {data_cfg_path}")
        with open(data_cfg_path, "w") as f:
            OmegaConf.save(config.dataset, f)

    return trainer
