from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset
import theseus as th
from matplotlib import pyplot as plt

from covpred.config.theseus.config import TheseusConfig
from covpred.config.evaluation import Config

# from covpred.config.training import Config
from covpred.evaluation.metrics.base_error import partial_rmse, r_t
from covpred.evaluation.metrics.metrics import rpe_1, rpe_n, l1_rpe_1, l1_rpe_n
from covpred.io.evaluation import save_metrics, save_live_methods, load_metrics, load_pre_computed_poses
from covpred.evaluation.frame_to_frame import Method, OptimizationFramework, frame_to_frame, color_and_linestyle
from covpred.dataset.precomp_matches import PreCompMatchesDataset
from covpred.math.klt_covariances import get_klt_unsc
from covpred.model.full_model import DeepPNEC, extract_covs
from covpred.common import (
    TranslationMode,
    custom_collate,
    get_entries_in_batch,
    get_rel_poses,
    rel_2_abs_poses,
    to_3d_cov,
    to_3d_point,
    translation_scale,
)
from covpred.math.projections import linear
from covpred.math.geometry import get_img_distances_to_model
from covpred.visualization.trajectories import Trajectory2D


@dataclass
class Result:
    poses: List[torch.Tensor]
    metrics: Optional[Dict[str, float]] = None


def get_uniform_covariances(image_points: torch.Tensor, K_inv: torch.Tensor) -> torch.Tensor:
    uniform_covs = torch.stack(2 * [torch.zeros_like(image_points)], dim=-1)
    uniform_covs[..., 0, 0] = 1.0
    uniform_covs[..., 1, 1] = 1.0
    bvs_uni_covs = linear(
        to_3d_point(image_points.flatten(0, 1)),
        K_inv,
        # get_entries_in_batch(K_inv, host_img_idx, batch_size),
        to_3d_cov(uniform_covs).type_as(image_points),
    )[1]
    return bvs_uni_covs


def get_reprojection_covariances(
    image_points: torch.Tensor, reprojection_error: torch.Tensor, K_inv: torch.Tensor
) -> torch.Tensor:
    isotropic_covs = torch.stack(2 * [torch.zeros_like(image_points)], dim=-1)
    isotropic_covs[..., 0, 0] = reprojection_error
    isotropic_covs[..., 1, 1] = reprojection_error
    bvs_repr_covs = linear(
        to_3d_point(image_points.flatten(0, 1)),
        K_inv,
        # get_entries_in_batch(K_inv, host_img_idx, batch_size),
        to_3d_cov(isotropic_covs).type_as(image_points),
    )[1]
    return bvs_repr_covs


def load_sequence_results(base_path: Path, method: Method) -> Optional[Result]:
    pre_comp_poses = load_pre_computed_poses(base_path, f"{method.name}.txt")
    if pre_comp_poses is None:
        return None
    pre_comp_metrics = load_metrics(base_path.joinpath(f"{method.name}_metrics.txt"))
    return Result(pre_comp_poses, pre_comp_metrics)


class MotionModel(Enum):
    NO = 0
    CONSTANT = 1
    GROUNDTRUTH = 2


def sequence_evaluation(
    methods: List[Method],
    dataset: PreCompMatchesDataset,
    model: DeepPNEC,
    config: Config,
    motion_model: MotionModel,
    translation_mode: TranslationMode,
    device: str,
    run_number: int,
    override_results: bool = False,
):
    matches_path = Path(config.dataset.directories.matches).joinpath(
        config.dataset.name, config.dataset.matching.algorithm
    )
    output_paths: Dict[Method, Path] = {}
    for method in methods:
        if method == Method.DEEP_PNEC:
            model_eval_path = Path(model.cfg.path).joinpath(
                model.cfg.base_model,
                model.cfg.date,
                dataset.dataset_config.name,
                dataset.dataset_config.matching.algorithm,
                "poses",
            )
            model_eval_path.mkdir(parents=True, exist_ok=True)
            output_paths[method] = model_eval_path
        else:
            matches_eval_path = Path(
                dataset.dataset_config.directories.matches,
                dataset.dataset_config.name,
                dataset.dataset_config.matching.algorithm,
                "poses",
            )
            matches_eval_path.mkdir(parents=True, exist_ok=True)
            output_paths[method] = matches_eval_path

    pose_est_fn = frame_to_frame(config, OptimizationFramework.CERES)
    with torch.no_grad():
        sequence_indices = dataset.get_sequence_indices()

        for sequence, indices in sequence_indices.items():
            image_paths = output_paths[Method.DEEP_PNEC].joinpath(f"run_{run_number}", sequence, "Images")
            image_paths.mkdir(parents=True, exist_ok=True)

            start = time.time()
            print(f"starting sequence {sequence}")
            # create pytorch subset
            sequence_subset = Subset(dataset, indices)
            sequence_loader = DataLoader(
                sequence_subset,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=custom_collate,
            )

            sequence_gt_poses: List[torch.Tensor] = [th.SE3().tensor[0]]
            sequence_results = {method: Result([th.SE3().tensor[0]]) for method in methods}

            # ==================================== LOAD METRICS AND POSES FROM PLATE ===================================
            if override_results:
                live_methods = methods
            else:
                live_methods: List[Method] = []
                for method in methods:
                    results = load_sequence_results(
                        output_paths[method].joinpath(f"run_{run_number}", sequence), method
                    )
                    if results is None:
                        live_methods.append(method)
                    else:
                        sequence_results[method] = results

            gt_path = Path(dataset.dataset_config.directories.matches).joinpath(sequence, f"poses_gt.txt")
            pre_comp_gt_poses = load_pre_computed_poses(matches_path.joinpath("poses", "gt_poses", sequence), f"gt.txt")
            has_gt = False
            if pre_comp_gt_poses:
                sequence_gt_poses = pre_comp_gt_poses
                print(f"loading {len(pre_comp_gt_poses)} precomputed ground truth poses from {gt_path}")
                has_gt = True

            # ================================================ POSE GEN ================================================
            if len(live_methods) > 0 or not has_gt:
                print(f"Calculating poses for {[method.name for method in live_methods]}")
                for batch_idx, data in enumerate(sequence_loader):
                    batch_size = data[0].shape[0]
                    # print(batch_idx)
                    images = data[0].to(device)  # B,N,1,H,W; torch.float32
                    K_inv = data[1].to(device)  # B,N,3,3; torch.float64
                    gt_poses = th.SE3(tensor=data[2].flatten(0, 1).to(device))  # B*N,3,4; torch.float64
                    host_keypoints = data[3].permute(0, 2, 1, 3).to(device)  # B,P,M,2
                    host_img_idx = data[4]  # B,P; torch.int32
                    target_keypoints = data[5].permute(0, 2, 1, 3).to(device)  # B,P,M,2
                    target_img_idx = data[6]  # B,P; torch.int32
                    masks = data[7].to(device).permute(0, 2, 1)  # B,P,M; torch.int32
                    confidences = data[8].to(device).permute(0, 2, 1)  # B,P,M; torch.float64

                    # gt_poses = get_rel_poses(poses, host_img_idx, target_img_idx, batch_size)  # B*P,3,4
                    host_K_inv = get_entries_in_batch(K_inv, host_img_idx, batch_size)
                    target_K_inv = get_entries_in_batch(K_inv, target_img_idx, batch_size)

                    host_bvs = linear(
                        to_3d_point(host_keypoints),
                        host_K_inv,
                    )[
                        0, 0
                    ]  # M,3
                    target_bvs = linear(
                        to_3d_point(target_keypoints),
                        target_K_inv,
                    )[
                        0, 0
                    ]  # M,3
                    host_covs: Dict[Method, torch.Tensor] = {}
                    target_covs: Dict[Method, torch.Tensor] = {}

                    if Method.DEEP_PNEC in live_methods:
                        cov_predictions = model(images)  # B*N,H,W,2,2
                        host_img_covs, target_img_covs = extract_covs(
                            cov_predictions, host_keypoints, host_img_idx, target_keypoints, target_img_idx
                        )  # B,P,M,2,2
                        host_img_covs = host_img_covs.to(torch.float64)
                        target_img_covs = target_img_covs.to(torch.float64)

                        host_bvs_covs = linear(
                            to_3d_point(host_keypoints[0, 0]), host_K_inv, to_3d_cov(host_img_covs[0, 0])
                        )[
                            1
                        ]  # M,3,3
                        target_bvs_covs = linear(
                            to_3d_point(target_keypoints[0, 0]), target_K_inv, to_3d_cov(target_img_covs[0, 0])
                        )[
                            1
                        ]  # M,3,3

                        host_covs[Method.DEEP_PNEC] = host_bvs_covs[0]
                        target_covs[Method.DEEP_PNEC] = target_bvs_covs[0]

                    if Method.KLT_PNEC in live_methods:
                        klt_covs = get_klt_unsc(images.flatten(0, 1))[:, 0]

                        host_img_covs, target_img_covs = extract_covs(
                            klt_covs, host_keypoints, host_img_idx, target_keypoints, target_img_idx
                        )
                        host_img_covs = host_img_covs.to(torch.float64)
                        target_img_covs = target_img_covs.to(torch.float64)

                        host_bvs_covs = linear(
                            to_3d_point(host_keypoints[0, 0]), host_K_inv, to_3d_cov(host_img_covs[0, 0])
                        )[1]
                        target_bvs_covs = linear(
                            to_3d_point(target_keypoints[0, 0]), target_K_inv, to_3d_cov(target_img_covs[0, 0])
                        )[1]

                        host_covs[Method.KLT_PNEC] = host_bvs_covs[0]
                        target_covs[Method.KLT_PNEC] = target_bvs_covs[0]

                    if Method.UNIFORM in live_methods:
                        host_covs[Method.UNIFORM] = get_uniform_covariances(host_keypoints, host_K_inv).flatten(0, 1)[0]
                        target_covs[Method.UNIFORM] = get_uniform_covariances(target_keypoints, target_K_inv).flatten(
                            0, 1
                        )[0]

                    if Method.REPROJECTION in live_methods:
                        host_repr_error, target_repr_error = get_img_distances_to_model(
                            host_bvs[None], target_bvs[None], gt_poses, K_inv[0, 0], K_inv[0, 0]
                        )
                        host_repr_error = torch.square(torch.clip(host_repr_error, 0.1, 2.0))
                        target_repr_error = torch.square(torch.clip(target_repr_error, 0.1, 2.0))
                        host_covs[Method.REPROJECTION] = get_reprojection_covariances(
                            host_keypoints[0], host_repr_error[0], host_K_inv
                        )[0]
                        target_covs[Method.REPROJECTION] = get_reprojection_covariances(
                            target_keypoints[0], target_repr_error[0], target_K_inv
                        )[0]

                    for idx in range(images.shape[0]):
                        if not has_gt:
                            sequence_gt_poses.append(gt_poses.tensor[idx].to("cpu"))

                        for method in live_methods:
                            init_pose = th.SE3()
                            init_pose.to(device)
                            if motion_model == MotionModel.CONSTANT:
                                init_pose = th.SE3(tensor=sequence_results[method].poses[-1][None].to(device))
                            if motion_model == MotionModel.GROUNDTRUTH:
                                init_pose = th.SE3(tensor=gt_poses.tensor[idx][None].to(device))
                            # if torch.abs(init_pose.tensor[0, :3, 3]).sum() == 0.0:
                            #     init_pose.tensor[0, :3, 3] = torch.tensor([0.0, 0.0, 1.0])

                            pose = pose_est_fn(
                                method,
                                host_bvs,  # M,3
                                target_bvs,  # M,3
                                init_pose,  # 1,3,4
                                host_covs.get(method, None),  # M,3,3
                                target_covs.get(method, None),  # M,3,3
                                confidences[0, 0],  # M
                            ).to("cpu")
                            sequence_results[method].poses.append(
                                translation_scale(pose, gt_poses.tensor[idx].to("cpu"), translation_mode)
                            )
            else:
                print(f"All poses come from plate")
            end = time.time()
            print(f"took {end - start} seconds")

            for method in live_methods:
                print(f"Saving {method.name} poses")
                save_live_methods(
                    output_paths[method].joinpath(f"run_{run_number}", sequence, f"{method.name}.txt"),
                    sequence_results[method].poses,
                )
                # save_live_methods(
                #     output_paths[method].joinpath(sequence, str(run_number), f"poses_{method.name}.txt"),
                #     sequence_results[method].poses,
                # )

            if not has_gt:
                print("Saving ground truth poses")
                save_live_methods(
                    gt_path,
                    sequence_gt_poses,
                )

            colors = []
            linestyles = []
            abs_gt_poses = rel_2_abs_poses(torch.stack(sequence_gt_poses)[1:])
            abs_est_poses: Dict[str, torch.Tensor] = {}
            for method, results in sequence_results.items():
                abs_est_poses[method.name] = rel_2_abs_poses(torch.stack(results.poses)[1:])
                color, linestyle = color_and_linestyle[method.value]
                colors.append(color)
                linestyles.append(linestyle)

            # ============================================== TRAJECTORIES ==============================================
            fig, ax = Trajectory2D(
                abs_gt_poses,
                ("x", "z"),
                abs_est_poses,
                colors,
                linestyles,
                (6, 6),
            )
            plt.savefig(image_paths.joinpath("trajectory_xz.pdf"))
            plt.close("all")

            fig, ax = Trajectory2D(
                abs_gt_poses,
                ("y", "z"),
                abs_est_poses,
                colors,
                linestyles,
                (6, 6),
            )
            plt.savefig(image_paths.joinpath("trajectory_yz.pdf"))
            plt.close("all")

            # =================================================== YPR ==================================================
            # TODO: YPR

            # ================================================= METRICS ================================================
            for method in methods:
                if sequence_results[method].metrics is None:
                    th_gt = th.SE3(tensor=abs_gt_poses.to(device))
                    th_est = th.SE3(tensor=abs_est_poses[method.name].to(device))
                    metrics = {
                        "rpe1": rpe_1(th_gt, th_est),
                        "rpen": rpe_n(th_gt, th_est),
                        "l1_rpe1": l1_rpe_1(th_gt, th_est),
                        "l1_rpen": l1_rpe_n(th_gt, th_est),
                        "r_t": r_t(th_gt, th_est),
                    }
                    sequence_results[method].metrics = metrics

                    save_metrics(
                        output_paths[method].joinpath(f"run_{run_number}", sequence, f"{method.name}_metrics.json"),
                        metrics,
                    )
                    # save_metrics(
                    #     output_paths[method].joinpath(sequence, str(run_number), f"metrics_{method.name}.txt"), metrics
                    # )

            all_metrics: Dict[str, Dict[str, float]] = {}
            for method, results in sequence_results.items():
                if results.metrics is not None:
                    all_metrics[method.name] = results.metrics

            with open(
                output_paths[Method.DEEP_PNEC].joinpath(f"run_{run_number}", sequence, "metrics.json"), "w"
            ) as fp:
                json.dump(all_metrics, fp, indent="\t")

            # ============================================== PARTIAL RMSE ==============================================
            # TODO: extra partial RMSE eval
