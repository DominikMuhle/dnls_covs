from datetime import datetime
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore

import torch
from torch.utils.tensorboard import SummaryWriter
import theseus as th
import theseus.constants

from covpred.common import DummySummaryWriter, load_output_filter
from covpred.config.model.config import ModelOutputConfig, FilterConfig
from covpred.synthetic.experiment_creation import create_problems, TrainingFrames
from covpred.dataset.synthetic_dataset import SyntheticDataset
from covpred.config.synthetic_config import SyntheticExperimentConfig
from covpred.training.synthetic import train
from covpred.model.parametrization import get_parametrization, CovarianceParameterization

FORCE_CPU = False
device = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"


cs = ConfigStore.instance()
cs.store(name="base_config", node=SyntheticExperimentConfig)


@hydra.main(version_base=None, config_path="../config", config_name="synthetic")
def main(cfg: SyntheticExperimentConfig):
    theseus.constants.EPS = cfg.theseus.EPS
    torch.manual_seed(0)
    batch_size = min(6400, (int)(cfg.synthetic.num_problems / 10))
    training_frames = TrainingFrames[cfg.synthetic.training_frames]

    # TODO: pass as hydra config
    # frame_pairs, gt_poses = create_problems(
    #     training_frames,
    #     cfg.synthetic.num_problems,
    #     cfg.synthetic.num_points,
    #     cfg.synthetic.individual_poses,
    #     cfg.synthetic.max_t,
    #     cfg.synthetic.max_r,
    # )
    frame_pairs, gt_poses = create_problems(cfg.synthetic)

    parametrization = get_parametrization(CovarianceParameterization.sab)
    output_filter = load_output_filter(
        ModelOutputConfig(
            filter1=FilterConfig(name="lukas"),
            filter2=FilterConfig(name="no"),
            filter3=FilterConfig(name="sigmoid"),
        )
    )

    dataset = SyntheticDataset(frame_pairs, gt_poses)

    log_dir = Path(cfg.synthetic.outdir).joinpath(cfg.synthetic.name, datetime.now().strftime("%Y_%m_%d_%H_%M"))
    log_dir.mkdir(parents=True, exist_ok=True)
    if cfg.logging.tensorboard.enabled:
        writer = SummaryWriter(
            log_dir=Path(cfg.logging.tensorboard.log_dir).joinpath(
                f'{cfg.synthetic.name}_{datetime.now().strftime("%Y_%m_%d_%H_%M")}'
            )
        )
    else:
        writer = DummySummaryWriter()

    # train(
    #     dataset,
    #     cfg.theseus,
    #     parametrization,
    #     output_filter,
    #     device,
    #     training_frames,
    #     log_dir,
    #     writer,
    #     cfg.synthetic.num_epochs,
    #     batch_size,
    # )

    train(
        dataset,
        cfg,
        parametrization,
        output_filter,
        device,
        training_frames,
        log_dir,
        writer,
        cfg.synthetic.num_epochs,
        batch_size,
    )


if __name__ == "__main__":
    main()
