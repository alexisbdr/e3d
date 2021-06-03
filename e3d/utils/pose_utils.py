import math

import numpy as np
import torch
from pytorch3d.transforms import so3_relative_angle
from pytorch3d.renderer import PerspectiveCameras


def qlog(q):
    """
    Applies logarithm map to q
    :param q: N x 4
    :return: N x 3
    """
    n = torch.norm(q[:, 1:], p=2, dim=1, keepdim=True)
    n = torch.clamp(n, min=1e-8)
    q = q[:, 1:] * torch.acos(torch.clamp(q[:, :1], min=-1.0, max=1.0))
    q = q / n
    return q


def qlog_numpy(q):
    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q


def qexp(q):
    """
    Applies the exponential map to q
    :param q: N x 3
    :return: N x 4
    """
    n = torch.norm(q, p=2, dim=1, keepdim=True)
    n = torch.clamp(n, min=1e-8)
    q = q * torch.sin(n)
    q = q / n
    q = torch.cat((torch.cos(n), q), dim=1)
    return q


def qexp_numpy(q):
    """
    Applies the exponential map to q
    :param q: (3,)
    :return: (4,)
    """
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n / np.pi) * q))
    return q


def quaternion_angular_error(q1, q2):

    d = abs(np.dot(q1, q2))
    theta = 2 * np.arccos(d) * 180 / math.pi
    return theta


def calc_vos_simple(poses):
    """
    calculate the VOs, from a list of consecutive poses
    :param poses: N x T x 7
    :return: N x (T-1) x 7
    """
    vos = []
    for p in poses:
        pvos = [p[i + 1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p) - 1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos


def calc_camera_distance(cam_1, cam_2):
    """
    Calculates the divergence of a batch of pairs of cameras cam_1, cam_2.
    The distance is composed of the cosine of the relative angle between
    the rotation components of the camera extrinsics and the l2 distance
    between the translation vectors.  -Pytorch3D
    """
    # rotation distance
    R_distance = (1.0 - so3_relative_angle(cam_1.R, cam_2.R, cos_angle=True)).mean()
    # translation distance
    T_distance = ((cam_1.T - cam_2.T) ** 2).sum(1).mean()
    # the final distance is the sum
    return R_distance + T_distance


def get_relative_camera(cams, edges):
    """
    For each pair of indices (i,j) in "edges" generate a camera
    that maps from the coordinates of the camera cams[i] to
    the coordinates of the camera cams[j] - Pytorch3D
    """

    # first generate the world-to-view Transform3d objects of each
    # camera pair (i, j) according to the edges argument
    trans_i, trans_j = [
        PerspectiveCameras(
            R=cams.R[edges[:, i]], T=cams.T[edges[:, i]], device=device,
        ).get_world_to_view_transform()
        for i in (0, 1)
    ]

    # compose the relative transformation as g_i^{-1} g_j
    trans_rel = trans_i.inverse().compose(trans_j)

    # generate a camera from the relative transform
    matrix_rel = trans_rel.get_matrix()
    cams_relative = PerspectiveCameras(
        R=matrix_rel[:, :3, :3], T=matrix_rel[:, 3, :3], device=device,
    )
    return cams_relative
