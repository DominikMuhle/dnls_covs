from datetime import datetime
import json
import logging
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from thirdparty.SuperGluePretrainedNetwork.models.matching import Matching

from covpred.dataset import Datasets, get_precomp_dataset
from covpred.config.evaluation import Config
from covpred.io.model import find_checkpoint_ignite, load_checkpoint_ignite, load_network_filters
from covpred.common import TranslationMode, get_largest_gpus, load_output_filter, model_info
import covpred.matching as matching
from covpred.model import get_model, Models
from covpred.model.full_model import DeepPNEC
from covpred.model.parametrization import (
    CovarianceParameterization,
    get_parametrization,
)
from covpred.evaluation.frame_to_frame import Method
from covpred.evaluation.sequence import MotionModel, sequence_evaluation

force_cpu = False

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base=None, config_path="../config", config_name="evaluation")
def main(cfg: Config):
    curr_time = datetime.now()
    output_path = Path(cfg.model.path).joinpath(cfg.model.base_model, cfg.model.date)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)12s %(levelname)8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=str(output_path.joinpath("evaluation.log")),
        filemode="w",
    )
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.INFO)
    logging.getLogger("ignite.engine.engine.Engine").name = "ignite.Engine"
    logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

    logger = logging.getLogger("evaluation")

    # run_nums = [4]
    run_nums = [0]
    # run_nums = [2, 3, 4]
    # run_nums = [0, 1, 2, 3, 4]

    cfg.dataset.sweep.skip = 0
    cfg.dataset.sweep.tuple_length = 2
    cfg.dataset.sweep.tuple_per_base_img = 1
    cfg.model.override_filters = False

    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"

    print('[INFO] Running inference on device "{}"'.format(device))

    # ===================================================== METHODS ====================================================
    methods = [Method[method] for method in cfg.methods]

    # ====================================================== MATCHING ======================================================
    matching_algorithm = matching.MatchingAlgorithm[cfg.dataset.matching.algorithm]
    if matching_algorithm == matching.MatchingAlgorithm.SUPERGLUE:
        matching_fn = matching.SuperGlue(Matching(cfg.matching).eval().to(device).to(torch.float32))
    if matching_algorithm == matching.MatchingAlgorithm.KLT:
        matching_fn = matching.KLT
    if matching_algorithm == matching.MatchingAlgorithm.ORB:
        matching_fn = matching.ORB

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

    to_load = {"model": model}
    ckp_path = find_checkpoint_ignite(model_cfg=cfg.model, logger=logger)
    if ckp_path:
        load_checkpoint_ignite(ckp_path, to_load, cfg.dataset)

    # ==================================================== DATASET =====================================================
    dataset_type = Datasets[cfg.dataset.name]
    eval_dataset = get_precomp_dataset(dataset_type)(
        cfg.dataset,
        cfg.pose_estimation.nec,
        "train",
        matching_fn,
        device,
        False,
    )

    override_results = False
    for run_num in run_nums:
        config_json = json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)
        config_json = "".join("\t" + line for line in config_json.splitlines(True))
        base_output_path = Path(model.cfg.path).joinpath(
            model.cfg.base_model, model.cfg.date, "evaluation", cfg.dataset.name
        )
        base_output_path.mkdir(parents=True, exist_ok=True)
        with open(base_output_path.joinpath("config.json"), "w") as fp:
            json.dump(config_json, fp)

        sequence_evaluation(
            methods=methods,
            dataset=eval_dataset,
            model=model,
            config=cfg,
            motion_model=MotionModel.CONSTANT,
            translation_mode=TranslationMode.GTLENGTH,
            device=device,
            run_number=run_num,
            override_results=override_results,
        )


if __name__ == "__main__":
    main()
