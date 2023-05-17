from enum import Enum
from functools import partial
from typing import Callable, Optional
import numpy as np

import torch
import theseus as th
import ignite.distributed as idist

import pyopengv
import pypnec
from covpred.config.theseus.config import TheseusConfig
from covpred.config.training import Config
from covpred.model.optimization.eigenvalue import nec
from covpred.model.optimization.theseus.optimization_layer import create_theseus_layer
from covpred.training.common import best_init_pose


class Method(Enum):
    NISTER = 0
    SEVENPT = 1
    EIGHTPT = 2
    STEWENIUS = 3
    NEC = 4
    NEC_LS = 5
    WEIGHTED_NEC = 6
    UNIFORM = 7
    KLT_PNEC = 8
    DEEP_PNEC = 9
    REPROJECTION = 10


color_and_linestyle = [
    ("r", "--"),
    ("r", "-."),
    ("r", "-."),
    ("r", ":"),
    ("b", "--"),
    ("b", "-."),
    ("b", ":"),
    ("tab:orange", ":"),
    ("tab:orange", "--"),
    ("tab:orange", "-."),
    ("g", "--"),
]


class OptimizationFramework(Enum):
    CERES = 1
    THESEUS = 2


def pypnec_ceres(config: Config) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, th.SE3], th.SE3]:
    # pnec_ceres = partial(pypnec.pyceres, regularization=config.pose_estimation.regularization)

    def _opt_fn(
        host_bvs: torch.Tensor,
        host_bvs_covs: torch.Tensor,
        target_bvs: torch.Tensor,
        target_bvs_covs: torch.Tensor,
        init_poses: th.SE3,
    ) -> th.SE3:
        poses = []
        for h_bvs, h_bvs_covs, t_bvs, t_bvs_covs, init_pose in zip(
            host_bvs, host_bvs_covs, target_bvs, target_bvs_covs, init_poses.tensor
        ):
            pose = np.eye(4)
            pose[:3, :] = init_pose.cpu().detach().numpy()
            poses.append(
                torch.from_numpy(
                    pypnec.pyceres(
                        h_bvs.cpu().detach().numpy(),
                        t_bvs.cpu().detach().numpy(),
                        h_bvs_covs.cpu().detach().numpy(),
                        t_bvs_covs.cpu().detach().numpy(),
                        pose,
                        config.pose_estimation.regularization,
                    )[:3, :]
                )
            )

        return th.SE3(tensor=torch.stack(poses, dim=0))

    return _opt_fn


def pynec_ceres() -> Callable[[torch.Tensor, torch.Tensor, th.SE3], th.SE3]:
    def _opt_fn(host_bvs: torch.Tensor, target_bvs: torch.Tensor, init_poses: th.SE3) -> th.SE3:
        poses = []
        for h_bvs, t_bvs, init_pose in zip(host_bvs, target_bvs, init_poses.tensor):
            pose = np.eye(4)
            pose[:3, :] = init_pose.cpu().detach().numpy()
            poses.append(
                torch.from_numpy(
                    pypnec.pyceresnec(h_bvs.cpu().detach().numpy(), t_bvs.cpu().detach().numpy(), pose)[:3, :]
                )
            )

        return th.SE3(tensor=torch.stack(poses, dim=0))

    return _opt_fn


def frame_to_frame(config: Config, opt_framework: OptimizationFramework):
    max_points = 1024
    # if opt_framework == OptimizationFramework.CERES:
    #     opt_fn = pypnec_ceres(config)
    #     opt_fn_nec_ls = pynec_ceres()
    # if opt_framework == OptimizationFramework.THESEUS:
    precision = torch.float64
    opt_fn = create_theseus_layer(
        config.theseus, config.pose_estimation, idist.device(), max_points, precision, True, False
    )

    opt_fn_nec_ls = create_theseus_layer(
        config.theseus, config.pose_estimation, idist.device(), max_points, precision, False, False
    )

    def _evaluate(
        method: Method,
        host_bvs: torch.Tensor,  # M,3
        target_bvs: torch.Tensor,  # M,3
        init_pose: th.SE3,  # 1,3,4
        host_bvs_covs: Optional[torch.Tensor],  # M,3,3
        target_bvs_covs: Optional[torch.Tensor],  # M,3,3
        weights: Optional[torch.Tensor],  # M
    ) -> torch.Tensor:
        init_rotation = init_pose.to_matrix()[0, :3, :3].to("cpu").detach().numpy()

        bvs_0 = host_bvs.cpu().detach().numpy()
        bvs_1 = target_bvs.cpu().detach().numpy()

        inlier_threshold = 8.0e-7

        if method in [Method.STEWENIUS, Method.NISTER, Method.SEVENPT, Method.EIGHTPT]:
            opengv_pose = torch.from_numpy(
                pyopengv.relative_pose_ransac_init(bvs_0, bvs_1, init_rotation, method.name, inlier_threshold)
            )
            return opengv_pose

        nec_pose, inlier = nec(host_bvs, target_bvs, init_pose, config.pose_estimation.nec)

        if (init_pose.tensor[0, :3, 3] == 0.0).sum() == 3:
            init_pose.tensor[0, :3, 3] = nec_pose.tensor[0, :3, 3]
        if method == Method.NEC:
            return nec_pose.tensor[0].to(host_bvs.dtype)

        if config.pose_estimation.nec.ransac:
            host_bvs = host_bvs[inlier == 1.0, :]
            target_bvs = target_bvs[inlier == 1.0, :]

            if host_bvs_covs is not None:
                host_bvs_covs = host_bvs_covs[inlier == 1.0, :, :]
            if target_bvs_covs is not None:
                target_bvs_covs = target_bvs_covs[inlier == 1.0, :, :]
            if weights is not None:
                weights = weights[inlier == 1.0]

        # pose_choice = [init_pose]
        pose_choice = [nec_pose, init_pose]
        nec_ls_init = best_init_pose(pose_choice, host_bvs, None, target_bvs, None, config.pose_estimation)
        # match opt_framework:
        #     case OptimizationFramework.CERES:
        #         nec_ls_pose = pynec_ceres()(
        #             host_bvs[None],
        #             target_bvs[None],
        #             nec_ls_init,
        #         )
        #     case OptimizationFramework.THESEUS:
        #         nec_ls_pose = opt_fn_nec_ls(
        #             host_bvs=host_bvs[None],
        #             target_bvs=target_bvs[None],
        #             init_poses=best_init,
        #         )[0]
        match opt_framework:
            case OptimizationFramework.CERES:
                bvs_covs_0 = torch.zeros_like(host_bvs)[..., None].repeat(1, 1, 3)
                bvs_covs_1 = torch.zeros_like(target_bvs)[..., None].repeat(1, 1, 3)

                nec_ls_pose = pypnec_ceres(config)(
                    host_bvs[None],
                    bvs_covs_0[None],
                    target_bvs[None],
                    bvs_covs_1[None],
                    nec_ls_init,
                )
            case OptimizationFramework.THESEUS:
                nec_ls_pose = opt_fn(
                    host_bvs[None],
                    None,
                    target_bvs[None],
                    None,
                    nec_ls_init,
                )[0]
        nec_ls_pose = th.SE3(tensor=nec_ls_pose.tensor.to(idist.device()))
        if method == Method.NEC_LS:
            return nec_ls_pose.tensor[0]

        pose_choice = [nec_ls_pose]

        if method == Method.WEIGHTED_NEC:
            best_init = best_init_pose(pose_choice, host_bvs, None, target_bvs, None, config.pose_estimation)
            match opt_framework:
                case OptimizationFramework.CERES:
                    return pynec_ceres()(
                        (host_bvs * weights[:, None])[None],
                        target_bvs[None],
                        best_init,
                    ).tensor[0]
                case OptimizationFramework.THESEUS:
                    return opt_fn_nec_ls(
                        host_bvs=(host_bvs * weights[:, None])[None],
                        target_bvs=target_bvs[None],
                        init_poses=best_init,
                    )[0].tensor[0]

        best_init = best_init_pose(
            pose_choice, host_bvs, host_bvs_covs, target_bvs, target_bvs_covs, config.pose_estimation
        )
        match opt_framework:
            case OptimizationFramework.CERES:
                return pypnec_ceres(config)(
                    host_bvs[None],
                    host_bvs_covs[None],
                    target_bvs[None],
                    target_bvs_covs[None],
                    best_init,
                ).tensor[0]
            case OptimizationFramework.THESEUS:
                return opt_fn(
                    host_bvs=host_bvs[None],
                    host_bvs_covs=host_bvs_covs[None],
                    target_bvs=target_bvs[None],
                    target_bvs_covs=target_bvs_covs[None],
                    init_poses=best_init,
                )[0].tensor[0]

        # if opt_framework == OptimizationFramework.CERES:
        #     if method == Method.WEIGHTED_NEC:
        #         return

        #     # if method == Method.NEC_LS:
        #     #     return pynec_ceres()(
        #     #         host_bvs[None],
        #     #         target_bvs[None],
        #     #         best_init,
        #     #     ).tensor[0]

        # if method == Method.WEIGHTED_NEC:

        # # if method == Method.NEC_LS:
        # #     return opt_fn_nec_ls(
        # #         host_bvs=host_bvs[None],
        # #         target_bvs=target_bvs[None],
        # #         init_poses=best_init,
        # #     )[
        # #         0
        # #     ].tensor[0]

    return _evaluate
