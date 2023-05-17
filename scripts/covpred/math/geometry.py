from typing import Tuple

import torch
import theseus as th


def triangulate_points(bvs_0: torch.Tensor, bvs_1: torch.Tensor, relative_pose: th.SE3) -> torch.Tensor:
    bvs_1_unrot = torch.einsum("ij,...j->...i", relative_pose.tensor[0, :3, :3], bvs_1)  # B*P,N,3
    b = torch.stack(
        [
            torch.einsum("i,...i->...", relative_pose.tensor[0, :3, 3], bvs_0),
            torch.einsum("i,...i->...", relative_pose.tensor[0, :3, 3], bvs_1_unrot),
        ],
        dim=-1,
    )  # B*P,N,2
    A_00 = torch.einsum("...i,...i->...", bvs_0, bvs_0)
    A_10 = torch.einsum("...i,...i->...", bvs_0, bvs_1_unrot)
    A_11 = -torch.einsum("...i,...i->...", bvs_1_unrot, bvs_1_unrot)
    A = torch.stack([A_00, -A_10, A_10, A_11], dim=-1).reshape(-1, 2, 2)  # B*P,2,2

    lam = torch.einsum("...ij,...j->...i", torch.linalg.inv(A), b)  # B*P,N,2
    xm = lam[..., 0][..., None] * bvs_0
    xn = lam[..., 1][..., None] * bvs_1 + relative_pose.tensor[0, :3, 3][None]

    return (xm + xn) / 2.0


def get_selected_distances_to_model(
    bvs_0: torch.Tensor, bvs_1: torch.Tensor, relative_pose: th.SE3
) -> Tuple[torch.Tensor, torch.Tensor]:
    norm_relative_pose = relative_pose.copy()  # B*P,3,4
    norm_relative_pose.tensor[0, :3, 3] = norm_relative_pose.tensor[0, :3, 3] / torch.linalg.norm(
        norm_relative_pose.tensor[0, :3, 3]
    )
    inv_norm_relative_pose = norm_relative_pose.inverse()

    triangulated_points = triangulate_points(bvs_0, bvs_1, norm_relative_pose)  # B*P,N,3
    reprojection_0 = triangulated_points
    repr_1 = []
    for tr_points, pose in zip(triangulated_points, inv_norm_relative_pose.tensor):
        repr_1.append(th.SE3(tensor=pose[None]).transform_from(tr_points).tensor)
    reprojection_1 = torch.stack(repr_1)
    # reprojection_1 = inv_norm_relative_pose.transform_from(triangulated_points)
    reprojection_0 = reprojection_0 / torch.linalg.norm(reprojection_0, dim=-1)[..., None]
    reprojection_1 = reprojection_1 / torch.linalg.norm(reprojection_1, dim=-1)[..., None]

    reprojection_error_0 = 1.0 - torch.einsum("B...i,B...i->B...", bvs_0, reprojection_0)
    reprojection_error_1 = 1.0 - torch.einsum("B...i,B...i->B...", bvs_1, reprojection_1)

    return reprojection_error_0, reprojection_error_1


def get_img_distances_to_model(
    bvs_0: torch.Tensor,
    bvs_1: torch.Tensor,
    relative_pose: th.SE3,
    K_inv_0: torch.Tensor,
    K_inv_1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    norm_relative_pose = relative_pose.copy()  # B*P,3,4
    norm_relative_pose.tensor[0, :3, 3] = norm_relative_pose.tensor[0, :3, 3] / torch.linalg.norm(
        norm_relative_pose.tensor[0, :3, 3]
    )
    inv_norm_relative_pose = norm_relative_pose.inverse()

    triangulated_points = triangulate_points(bvs_0, bvs_1, norm_relative_pose)  # B*P,N,3
    reprojection_0 = triangulated_points
    repr_1 = []
    for tr_points, pose in zip(triangulated_points, inv_norm_relative_pose.tensor):
        repr_1.append(th.SE3(tensor=pose[None]).transform_from(tr_points).tensor)
    reprojection_1 = torch.stack(repr_1)
    reprojection_0 = reprojection_0 / reprojection_0[..., 2][..., None]
    reprojection_1 = reprojection_1 / reprojection_1[..., 2][..., None]
    K_0 = torch.linalg.inv(K_inv_0)
    K_1 = torch.linalg.inv(K_inv_1)
    repr_img_pts_0 = torch.einsum("ij,...j->...i", K_0, reprojection_0)[..., :2]
    repr_img_pts_1 = torch.einsum("ij,...j->...i", K_1, reprojection_1)[..., :2]

    proj_0 = bvs_0 / bvs_0[..., 2][..., None]
    proj_1 = bvs_1 / bvs_1[..., 2][..., None]
    img_pts_0 = torch.einsum("ij,...j->...i", K_0, proj_0)[..., :2]
    img_pts_1 = torch.einsum("ij,...j->...i", K_1, proj_1)[..., :2]

    return (
        torch.linalg.norm((img_pts_0 - repr_img_pts_0), dim=-1),
        torch.linalg.norm((img_pts_1 - repr_img_pts_1), dim=-1),
    )
