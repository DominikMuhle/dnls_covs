from datetime import datetime
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset
import torch.nn as nn
import torch.optim as optim
import hydra
from hydra.core.config_store import ConfigStore

from thirdparty.SuperGluePretrainedNetwork.models.matching import Matching
from covpred.common import (
    custom_collate,
    get_largest_gpus,
    load_output_filter,
    model_info,
)
import covpred.matching as matching
from covpred.config.training import Config

from covpred.model import get_model, Models
from covpred.model.full_model import DeepPNEC
from covpred.dataset import Datasets, get_live_dataset, get_precomp_dataset
from covpred.model.parametrization import (
    CovarianceParameterization,
    get_parametrization,
)
from covpred.training.training import ignite_training

FORCE_CPU = False
OVERRIDE_MATCHES = False

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base=None, config_path="../config", config_name="kitti")
def main(cfg: Config):
    # ===================================================== LOGGING ====================================================
    # TODO: ignite.engine.engine.Engine for console only WARNINGS
    curr_time = datetime.now()
    output_path = Path(cfg.model.path).joinpath(cfg.model.base_model, curr_time.strftime("%Y_%m_%d_%H_%M"))
    output_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)12s %(levelname)8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=str(output_path.joinpath("training.log")),
        filemode="w",
    )
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.INFO)
    logging.getLogger("ignite.engine.engine.Engine").name = "ignite.Engine"
    logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)14s: %(levelname)8s %(message)s")
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger("").addHandler(console)

    device = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
    logging.info(f"Running inference on device {device}")

    # ==================================================== MATCHING ====================================================
    matching_algorithm = matching.MatchingAlgorithm[cfg.dataset.matching.algorithm]
    match matching_algorithm:
        case matching.MatchingAlgorithm.SUPERGLUE:
            matching_fn = matching.SuperGlue(Matching(cfg.matching).eval().to(device).to(torch.float32))
        case matching.MatchingAlgorithm.KLT:
            matching_fn = matching.KLT
        case matching.MatchingAlgorithm.ORB:
            matching_fn = matching.ORB

    # ==================================================== DATASET =====================================================
    dataset_type = Datasets[cfg.dataset.name]
    training_dataset = get_live_dataset(dataset_type)(cfg.dataset, "train", device)
    eval_datasets = {
        "Training": get_precomp_dataset(dataset_type)(
            cfg.dataset,
            cfg.pose_estimation.nec,
            "train",
            matching_fn,
            device,
            OVERRIDE_MATCHES,
        ),
        "Validation": get_precomp_dataset(dataset_type)(
            cfg.dataset,
            cfg.pose_estimation.nec,
            "val",
            matching_fn,
            device,
            OVERRIDE_MATCHES,
        ),
    }

    pin_memory = True
    train_loader = DataLoader(
        training_dataset,
        batch_size=cfg.hyperparameter.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=pin_memory,
        collate_fn=custom_collate,
    )
    eval_loaders = {
        name: DataLoader(
            dataset,
            batch_size=2 * cfg.hyperparameter.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate,
        )
        for name, dataset in eval_datasets.items()
    }
    vis_loader = DataLoader(
        Subset(eval_datasets[sorted(eval_datasets.keys())[0]], [0, -1]),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=custom_collate,
    )

    # ===================================================== MODEL ======================================================
    unsc_net = get_model(Models[cfg.model.base_model])
    if device == "cpu":
        unsc_net = unsc_net.to(device)
    else:
        largest_gpus = get_largest_gpus()
        if len(largest_gpus) > 1:
            logging.info(f"Found {len(largest_gpus)} of the largest gpus")
            cfg.hyperparameter.batch_size *= len(largest_gpus)
            logging.info(
                f"Increasing batch size to {cfg.hyperparameter.batch_size * cfg.hyperparameter.accumulation_steps}"
            )
            unsc_net = nn.DataParallel(unsc_net, largest_gpus)
        unsc_net.to(device)

    output_filter = load_output_filter(cfg.model.output)
    representation = get_parametrization(CovarianceParameterization[cfg.model.output.representation])
    model = DeepPNEC(unsc_net, output_filter, representation, cfg.model)
    model_info(unsc_net)

    # =============================================== LOSS AND OPTIMIZER ===============================================
    loss_function = torch.nn.L1Loss()
    optimizer = optim.Adam(unsc_net.parameters(), lr=cfg.hyperparameter.learning_rate)

    lr_scheduler = optim.lr_scheduler.ChainedScheduler(
        [
            optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0 / cfg.hyperparameter.batch_size, end_factor=1.0, total_iters=50
            ),  # warmup
            optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99995),  # lr decay
        ]
    )

    ignite_training(
        train_loader,
        eval_loaders,
        vis_loader,
        matching_fn,
        model,
        optimizer,
        lr_scheduler,
        loss_function,
        cfg,
        output_path,
        curr_time,
    )


if __name__ == "__main__":
    main()
