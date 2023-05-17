from math import sqrt
import copy
from typing import Callable, Optional, Tuple

import theseus as th
from theseus.optimizer.nonlinear.nonlinear_optimizer import NonlinearOptimizerInfo, NonlinearOptimizer
import torch
from covpred.config.pose_estimation.config import PoseEstimationConfig

from covpred.math.common import skew, carthesian_to_sphere, sphere_to_carthesian, angular_diff, translational_diff
from covpred.math.pose_optimization import denominator
from covpred.config.theseus.config import TheseusConfig
from covpred.model.optimization.theseus.nec import NECCost
from covpred.model.optimization.theseus.pnec import PNECCost


def end_iter_callback(
    optimizer: NonlinearOptimizer, info: NonlinearOptimizerInfo, delta: torch.Tensor, it: int, **kwargs
):
    scalar = 2.0
    inv_scalar = 1 / scalar
    if info.err_history is not None and "damping" in kwargs:
        if torch.any(info.err_history[:, it] > info.err_history[:, it - 1]):
            kwargs["damping"] = kwargs["damping"] * scalar
        else:
            kwargs["damping"] = kwargs["damping"] * inv_scalar
    return kwargs


# def theseus_opt(
#     host_bvs: torch.Tensor,
#     host_bvs_covs: Optional[torch.Tensor],
#     target_bvs: torch.Tensor,
#     target_bvs_covs: Optional[torch.Tensor],
#     init_poses: th.SE3,
#     theseus_config: TheseusConfig,
#     pose_estimation_config: PoseEstimationConfig,
#     device: torch.device,
#     precision: torch.dtype,
#     only_r: bool = False,
# ) -> Tuple[th.SE3, NonlinearOptimizerInfo]:
#     pnec = (host_bvs_covs is not None) or (target_bvs_covs is not None)

#     init = th.SE3(tensor=copy.deepcopy(init_poses.tensor))
#     # init = init_poses

#     # TODO: remove explicit precision from function
#     host_bvs = host_bvs.to(precision)
#     if host_bvs_covs is not None:
#         host_bvs_covs = host_bvs_covs.to(precision)
#     target_bvs = target_bvs.to(precision)
#     if target_bvs_covs is not None:
#         target_bvs_covs = target_bvs_covs.to(precision)
#     init_poses.to(precision)

#     nec_scaling = 1.0
#     if not pnec:
#         nec_scaling = 1.0e7

#     cost_weight = th.ScaleCostWeight(th.Variable(torch.tensor([[1.0]], dtype=precision)))
#     rotations = th.SO3(
#         tensor=init_poses.tensor[..., :3, :3],
#         name="rotations",
#     )
#     translations = th.Vector(tensor=carthesian_to_sphere(init_poses.tensor[..., :3, 3]), name="translations")
#     bvs_0_hat = th.Variable(
#         tensor=skew(host_bvs),
#         name="bvs_0_hat",
#     )
#     bvs_1 = th.Variable(
#         tensor=target_bvs,
#         name="bvs_1",
#     )
#     theseus_inputs = {
#         "rotations": init_poses.tensor[..., :3, :3],
#         "translations": carthesian_to_sphere(init_poses.tensor[..., :3, 3]),
#         "bvs_0_hat": skew(host_bvs),
#         "bvs_1": target_bvs,
#         "scaling": torch.ones(
#             init_poses.tensor.shape[0], 1, dtype=init_poses.tensor.dtype, device=init_poses.tensor.device
#         )
#         * nec_scaling,
#     }
#     scaling = th.Variable(
#         tensor=pose_estimation_config.scaling
#         * torch.ones(init_poses.tensor.shape[0], 1, dtype=init_poses.tensor.dtype, device=init_poses.tensor.device)
#         * nec_scaling,
#         name="scaling",
#     )
#     regularization = th.Variable(
#         tensor=pose_estimation_config.regularization
#         * torch.ones(init_poses.tensor.shape[0], 1, dtype=init_poses.tensor.dtype, device=init_poses.tensor.device),
#         name="regularization",
#     )

#     if pnec:
#         zero_covariances = torch.zeros(
#             host_bvs.shape[:-1]
#             + (
#                 3,
#                 3,
#             ),
#             dtype=init_poses.tensor.dtype,
#             device=init_poses.tensor.device,
#         )
#         if host_bvs_covs is not None:
#             covariances_0 = th.Variable(
#                 tensor=host_bvs_covs,
#                 name="covariances_0",
#             )
#             theseus_inputs["covariances_0"] = host_bvs_covs
#         else:
#             covariances_0 = th.Variable(tensor=zero_covariances)
#             theseus_inputs["covariances_0"] = zero_covariances
#         if target_bvs_covs is not None:
#             covariances_1 = th.Variable(
#                 tensor=target_bvs_covs,
#                 name="covariances_1",
#             )
#             theseus_inputs["covariances_1"] = target_bvs_covs
#         else:
#             covariances_1 = th.Variable(tensor=zero_covariances)
#             theseus_inputs["covariances_1"] = zero_covariances

#         cost_fn = PNECCost(
#             cost_weight,
#             rotations,
#             translations,
#             bvs_0_hat,
#             bvs_1,
#             covariances_0,
#             covariances_1,
#             regularization=regularization,
#             scaling=scaling,
#             name="pnec",
#             opt_t=not only_r,
#         )
#     else:
#         cost_fn = NECCost(
#             cost_weight,
#             rotations,
#             translations,
#             bvs_0_hat,
#             bvs_1,
#             scaling=scaling,
#             name="nec",
#             opt_t=not only_r,
#         )

#     if device == "cpu":
#         linearization_cls = th.DenseLinearization
#         linear_solver_cls = th.CholeskyDenseSolver
#     else:
#         linearization_cls = th.SparseLinearization
#         linear_solver_cls = th.LUCudaSparseSolver

#     objective = th.Objective(dtype=precision)
#     objective.add(cost_fn)
#     objective.to(device)

#     dnls_optimizer = th.LevenbergMarquardt(
#         objective,
#         linearization_cls=linearization_cls,
#         linear_solver_cls=linear_solver_cls,
#         vectorize=True,
#         empty_cuda_cache=True,
#         step_size=theseus_config.step_size,
#         abs_err_tolerance=theseus_config.abs_err_tolerance,
#         rel_err_tolerance=theseus_config.rel_err_tolerance,
#     )

#     theseus_layer = th.TheseusLayer(dnls_optimizer)
#     theseus_layer.to(device)

#     if isinstance(theseus_config.optimizer.kwargs.get("backward_mode"), str):
#         theseus_config.optimizer.kwargs["backward_mode"] = th.BackwardMode[
#             theseus_config.optimizer.kwargs["backward_mode"]
#         ]

#     # updated_inputs, _ = theseus_layer(
#     #     theseus_inputs,
#     #     optimizer_kwargs={key: value for key, value in theseus_config.optimizer.kwargs.items()},
#     # )
#     # if cost_fn.opt_R:
#     #     final_rotations = updated_inputs["rotations"]
#     # else:
#     #     final_rotations = init[..., :3, :3]
#     # if cost_fn.opt_t:
#     #     final_translations, _ = sphere_to_carthesian(updated_inputs["translations"])
#     # else:
#     #     final_translations = init[..., :3, 3]
#     # return th.SE3(
#     #     tensor=torch.concat(
#     #         [final_rotations, final_translations[..., :, None]],
#     #         dim=-1,
#     #     )
#     # )

#     updated_inputs, info = theseus_layer(
#         theseus_inputs,
#         optimizer_kwargs={key: value for key, value in theseus_config.optimizer.kwargs.items()},
#     )

#     # TODO: integrate into a better logging system
#     diverged = (info.last_err.to("cpu") - info.err_history[..., 0] > 0).sum()
#     num_optimizations = info.last_err.size()[0]
#     if diverged:
#         print(f"[Warning]: Out of {num_optimizations} optimizations {diverged.item()} diverged")

#     if cost_fn.opt_R:
#         final_rotations = updated_inputs["rotations"]
#     else:
#         final_rotations = init_poses.tensor[..., :3, :3]
#     if cost_fn.opt_t:
#         final_translations, _ = sphere_to_carthesian(updated_inputs["translations"])
#     else:
#         final_translations = init_poses.tensor[..., :3, 3]
#     return (
#         th.SE3(
#             tensor=torch.concat(
#                 [final_rotations, final_translations[..., :, None]],
#                 dim=-1,
#             )
#         ),
#         info,
#     )


TheseusLayer = Callable[
    [
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        th.SE3,
    ],
    Tuple[th.SE3, NonlinearOptimizerInfo],
]


def create_theseus_layer(
    theseus_config: TheseusConfig,
    pose_estimation_config: PoseEstimationConfig,
    device: torch.device,
    max_points: int,
    precision: torch.dtype,
    pnec: bool,
    only_r: bool = False,
) -> TheseusLayer:
    nec_scaling = 1.0
    if not pnec:
        nec_scaling = 1.0e7

    cost_weight = th.ScaleCostWeight(th.Variable(torch.tensor([[1.0]], dtype=precision)))
    rotations = th.SO3(
        tensor=torch.eye(3, dtype=precision, device=device)[None],
        name="rotations",
    )
    translations = th.Vector(tensor=torch.zeros(1, 2, dtype=precision, device=device), name="translations")
    bvs_0_hat = th.Variable(
        tensor=torch.zeros(1, max_points, 3, 3, dtype=precision, device=device),
        name="bvs_0_hat",
    )
    bvs_1 = th.Variable(
        tensor=torch.zeros(1, max_points, 3, dtype=precision, device=device),
        name="bvs_1",
    )
    scaling = th.Variable(
        tensor=pose_estimation_config.scaling * torch.ones(1, 1, dtype=precision, device=device) * nec_scaling,
        name="scaling",
    )
    regularization = th.Variable(
        tensor=pose_estimation_config.regularization * torch.ones(1, 1, dtype=precision, device=device),
        name="regularization",
    )
    covariances_0 = th.Variable(
        tensor=torch.zeros(1, max_points, 3, 3, dtype=precision, device=device),
        name="covariances_0",
    )
    covariances_1 = th.Variable(
        tensor=torch.zeros(1, max_points, 3, 3, dtype=precision, device=device),
        name="covariances_1",
    )
    if pnec:
        cost_fn = PNECCost(
            cost_weight,
            rotations,
            translations,
            bvs_0_hat,
            bvs_1,
            covariances_0,
            covariances_1,
            regularization=regularization,
            scaling=scaling,
            name="pnec",
            opt_t=not only_r,
        )
    else:
        cost_fn = NECCost(
            cost_weight,
            rotations,
            translations,
            bvs_0_hat,
            bvs_1,
            scaling=scaling,
            name="nec",
            opt_t=not only_r,
        )

    if device == "cpu":
        linearization_cls = th.DenseLinearization
        linear_solver_cls = th.CholeskyDenseSolver
    else:
        linearization_cls = th.SparseLinearization
        linear_solver_cls = th.LUCudaSparseSolver

    objective = th.Objective(dtype=precision)
    objective.add(cost_fn)
    objective.to(device)

    dnls_optimizer = th.LevenbergMarquardt(
        objective,
        linearization_cls=linearization_cls,
        linear_solver_cls=linear_solver_cls,
        vectorize=True,
        empty_cuda_cache=True,
        step_size=theseus_config.step_size,
        abs_err_tolerance=theseus_config.abs_err_tolerance,
        rel_err_tolerance=theseus_config.rel_err_tolerance,
        **theseus_config.optimizer.kwargs,
    )

    theseus_layer = th.TheseusLayer(dnls_optimizer)
    theseus_layer.to(device)

    optimizer_kwargs = theseus_config.optimizer.kwargs.copy()

    if isinstance(optimizer_kwargs.get("backward_mode"), str):
        optimizer_kwargs["backward_mode"] = th.BackwardMode[optimizer_kwargs["backward_mode"]]

    def theseus_opt(
        host_bvs: torch.Tensor,
        host_bvs_covs: Optional[torch.Tensor],
        target_bvs: torch.Tensor,
        target_bvs_covs: Optional[torch.Tensor],
        init_poses: th.SE3,
    ) -> Tuple[th.SE3, NonlinearOptimizerInfo]:
        pnec = (host_bvs_covs is not None) or (target_bvs_covs is not None)

        # init = th.SE3(tensor=copy.deepcopy(init_poses.tensor))
        # init = init_poses

        # TODO: remove explicit precision from function
        host_bvs = host_bvs.to(precision)
        if host_bvs_covs is not None:
            host_bvs_covs = host_bvs_covs.to(precision)
        target_bvs = target_bvs.to(precision)
        if target_bvs_covs is not None:
            target_bvs_covs = target_bvs_covs.to(precision)
        init_poses.to(precision)

        theseus_inputs = {
            "rotations": init_poses.tensor[..., :3, :3],
            "translations": carthesian_to_sphere(init_poses.tensor[..., :3, 3]),
            "bvs_0_hat": skew(
                torch.nn.functional.pad(host_bvs, (0, 0, 0, max_points - host_bvs.shape[-2]), "constant", 1.0 / sqrt(3))
            ),
            "bvs_1": torch.nn.functional.pad(target_bvs, (0, 0, 0, max_points - target_bvs.shape[-2]), "constant", 0.0),
            "scaling": torch.ones(
                init_poses.tensor.shape[0], 1, dtype=init_poses.tensor.dtype, device=init_poses.tensor.device
            )
            * nec_scaling,
        }

        if pnec:
            zero_covariances = torch.zeros(
                host_bvs.shape[:-2]
                + (
                    max_points,
                    3,
                    3,
                ),
                dtype=init_poses.tensor.dtype,
                device=init_poses.tensor.device,
            )
            if host_bvs_covs is not None:
                theseus_inputs["covariances_0"] = torch.nn.functional.pad(
                    host_bvs_covs, (0, 0, 0, 0, 0, max_points - host_bvs_covs.shape[-3]), "constant", 0.0
                )
            else:
                theseus_inputs["covariances_0"] = zero_covariances
            if target_bvs_covs is not None:
                theseus_inputs["covariances_1"] = torch.nn.functional.pad(
                    target_bvs_covs, (0, 0, 0, 0, 0, max_points - target_bvs_covs.shape[-3]), "constant", 0.0
                )
            else:
                theseus_inputs["covariances_1"] = zero_covariances

        updated_inputs, info = theseus_layer(
            theseus_inputs,
            optimizer_kwargs={key: value for key, value in optimizer_kwargs.items()},
        )

        # TODO: integrate into a better logging system
        diverged = (info.last_err.to("cpu") - info.err_history[..., 0] > 0).sum()
        num_optimizations = info.last_err.size()[0]
        if diverged:
            print(f"[Warning]: Out of {num_optimizations} optimizations {diverged.item()} diverged")

        if cost_fn.opt_R:
            final_rotations = updated_inputs["rotations"]
        else:
            final_rotations = init_poses.tensor[..., :3, :3]
        if cost_fn.opt_t:
            final_translations, _ = sphere_to_carthesian(updated_inputs["translations"])
        else:
            final_translations = init_poses.tensor[..., :3, 3]
        return (
            th.SE3(
                tensor=torch.concat(
                    [final_rotations, final_translations[..., :, None]],
                    dim=-1,
                )
            ),
            info,
        )

    return theseus_opt
