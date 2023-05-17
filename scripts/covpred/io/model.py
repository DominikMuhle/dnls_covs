from pathlib import Path
from typing import Any, Dict, Optional
from logging import Logger

import torch
from omegaconf import OmegaConf
from ignite.handlers import Checkpoint

from covpred.common import load_output_filter
from covpred.config.dataset.config import DatasetConfig
from covpred.config.model.config import ModelConfig, ModelOutputConfig
from covpred.model.output_filter import OutputFilter


def find_checkpoint_ignite(model_cfg: ModelConfig, logger: Logger) -> Optional[Path]:
    base_path = Path(model_cfg.path)

    # Find version
    version_path = base_path.joinpath(model_cfg.base_model)
    if not version_path.is_dir():
        logger.warning("No model found, starting with fresh model")
        return None

    # Find specified date
    # get the latest date as default path, override if specified date is available
    date_path = [path for path in sorted(version_path.iterdir()) if path.is_dir()][-1]
    if model_cfg.date is None:
        logger.warning(f"No date specified, loading the latest model")
        model_cfg.checkpoint = None
    else:
        if version_path.joinpath(model_cfg.date).is_dir():
            date_path = version_path.joinpath(model_cfg.date)
        else:
            logger.warning(f"Specified date not found, loading the latest model")
            model_cfg.checkpoint = None

    model_path = date_path.joinpath(f"finished")
    if model_cfg.checkpoint is not None:
        model_path = date_path.joinpath(model_cfg.checkpoint)
    if not model_path.is_file():
        logger.warning(f"Specified checkpoint not found, loading the latest checkpoint")
        models = [model for model in sorted(date_path.iterdir()) if model.suffix == ".pt"]
        model_path = models[-1]

    return model_path


def load_checkpoint_ignite(model_path: Path, to_save: Dict[str, Any], dataset_cfg: Optional[DatasetConfig] = None):
    checkpoint = torch.load(model_path.as_posix(), map_location="cpu")
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    if dataset_cfg is not None:
        stored_config_path = model_path.parent.joinpath("dataset.yaml")
        if not stored_config_path.is_file():
            print(f"[INFO] No dataset config file found. Continuing with given dataset config.")
        else:
            stored_config = OmegaConf.load(stored_config_path)
            print(dataset_cfg.matching.algorithm)
            print(stored_config.matching.algorithm)
            print(f"[INFO] Updating the dataset config with the model specific dataset matching algorithm.")
            dataset_cfg.directories.matches = stored_config.directories.matches
            dataset_cfg.matching.algorithm = stored_config.matching.algorithm


def load_network_filters(model_path: Path, model_cfg: ModelConfig) -> OutputFilter:
    if model_cfg.override_filters:
        return load_output_filter(model_cfg.output)

    stored_config_path = model_path.parent.joinpath("model_output.yaml")
    if not stored_config_path.is_file():
        print(f"[INFO] No output config file found. Continuing with given output config.")
        return load_output_filter(model_cfg.output)

    stored_config: ModelOutputConfig = OmegaConf.load(stored_config_path)
    model_cfg.output = stored_config
    return load_output_filter(stored_config)