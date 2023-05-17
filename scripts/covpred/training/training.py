import json
from pathlib import Path
from typing import Dict
from datetime import datetime
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
import ignite.distributed as idist
from omegaconf import OmegaConf

from covpred.config.training import Config
import covpred.matching as matching
from covpred.model.full_model import DeepPNEC
from covpred.model.optimization.theseus.optimization_layer import create_theseus_layer
from covpred.training.epoch import create_trainer, visualize_training_examples
from covpred.training.evaluation import create_evaluator, create_visualization_evaluator


def ignite_training(
    train_loader: DataLoader,
    eval_loaders: Dict[str, DataLoader],
    vis_loader: DataLoader,
    matching_fn: matching.MatchingFunction,
    model: DeepPNEC,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    loss_function: torch.nn.modules.loss._Loss,
    config: Config,
    output_path: Path,
    time: datetime,
):
    # Define a Tensorboard logger
    log_dir = Path(config.logging.tensorboard.log_dir).joinpath(
        f"{config.logging.tensorboard.name}_{time.strftime('%Y_%m_%d_%H_%M')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=log_dir)

    logger = logging.getLogger("training")

    precision = torch.float64
    theseus_opt = create_theseus_layer(
        config.theseus, config.pose_estimation, idist.device(), 512, precision, True, False
    )

    visualize_training_examples(train_loader.dataset, matching_fn, 0, 5, tb_logger, logger)
    trainer = create_trainer(
        model,
        matching_fn,
        theseus_opt,
        config,
        loss_function,
        optimizer,
        lr_scheduler,
        output_path,
        logger,
        tb_logger,
    )

    config_json = json.dumps(OmegaConf.to_container(config, resolve=True), indent=2)
    config_json = "".join("\t" + line for line in config_json.splitlines(True))
    tb_logger.writer.add_text("config", text_string=config_json, global_step=0)

    evaluators: Dict[str, Engine] = {
        name: create_evaluator(model=model, theseus_opt=theseus_opt, config=config) for name in eval_loaders.keys()
    }
    vis_evaluator = create_visualization_evaluator(model, config, tb_logger, trainer)

    val_evaluator = evaluators.get("Validation", None)
    if val_evaluator is None:
        val_evaluator = evaluators[sorted(evaluators.keys())[-1]]

    # @trainer.on(Events.EPOCH_COMPLETED(every=config.logging.tensorboard.eval_epochs))
    # @trainer.on(Events.EPOCH_COMPLETED(every=2))
    @trainer.on(Events.STARTED | Events.EPOCH_COMPLETED(every=config.logging.tensorboard.eval_epochs))
    def log_training_results(trainer):
        for name, evaluator in evaluators.items():
            logger.debug(f"Running evaluation for {name}")
            evaluator.run(eval_loaders[name])
            metrics = evaluator.state.metrics
            metrics_str = ""
            for m_name, metric in metrics.items():
                metrics_str += f"{m_name}: {metric:.3f} "
            logger.info(f"{name} Results - Epoch[{trainer.state.epoch}] | {metrics_str}")

    @trainer.on(Events.STARTED | Events.EPOCH_COMPLETED(every=config.logging.tensorboard.vis_epochs))
    def viz_covariances(trainer):
        logger.debug(f"Create visualization of covariances.")
        vis_evaluator.run(vis_loader)

    # Score function to return current value of any metric we defined above in val_metrics
    def score_function(engine):
        return engine.state.metrics["rotational error"]

    # Checkpoint to store n_saved best models wrt score function
    best_checkpoint = ModelCheckpoint(
        dirname=output_path,
        n_saved=2,
        filename_prefix="best",
        score_function=score_function,
        score_name="rot_error",
        global_step_transform=global_step_from_engine(trainer),  # helps fetch the trainer's state
    )

    # Checkpoint to store n_saved best models wrt score function
    save_epoch_checkpoint = ModelCheckpoint(
        dirname=output_path,
        n_saved=None,
        filename_prefix="epoch",
        global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
    )

    val_evaluator.add_event_handler(
        Events.STARTED | Events.EPOCH_COMPLETED(once=1) | Events.EPOCH_COMPLETED(every=5) | Events.TERMINATE,
        save_epoch_checkpoint,
        {"model": model},
    )

    # # Save the model after every epoch of val_evaluator is completed
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_checkpoint, {"model": model})

    # Attach handler for plotting both evaluators' metrics after every epoch completes
    for tag, evaluator in evaluators.items():
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    trainer.run(train_loader, max_epochs=config.hyperparameter.epochs)
