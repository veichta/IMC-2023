
import torch
import numpy as np

def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points, eps=0.):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)


def sym_epipolar_distance_all(p0, p1, E, eps=1e-15):
    if p0.shape[-1] != 3:
        p0 = to_homogeneous(p0)
    if p1.shape[-1] != 3:
        p1 = to_homogeneous(p1)
    p1_E_p0 = torch.einsum('...mi,...ij,...nj->...nm', p1, E, p0).abs()
    E_p0 = torch.einsum('...ij,...nj->...ni', E, p0)
    Et_p1 = torch.einsum('...ij,...mi->...mj', E, p1)
    d0 = p1_E_p0 / (
        E_p0[..., None, 0] ** 2 + E_p0[..., None, 1]**2 + eps).sqrt()
    d1 = p1_E_p0 / (
        Et_p1[..., None, :, 0] ** 2 + Et_p1[..., None, :, 1]**2 + eps).sqrt()
    return (d0 + d1) / 2


def warp_points_torch(points, H, inverse=True):
    points = to_homogeneous(points)

    # Apply the homography
    H_mat = (torch.inverse(H) if inverse else H).transpose(-2, -1)
    warped_points = torch.einsum('...nj,...ji->...ni', points, H_mat)

    warped_points = from_homogeneous(warped_points, eps=1e-5)
    return warped_points


def sym_homography_error_all(kpts0, kpts1, H):
    kp0_1 = warp_points_torch(kpts0, H, inverse=False)
    kp1_0 = warp_points_torch(kpts1, H, inverse=True)

    # build a distance matrix of size [... x M x N]
    dist0 = torch.sum((kp0_1.unsqueeze(-2) - kpts1.unsqueeze(-3)) ** 2, -1).sqrt()
    dist1 = torch.sum((kpts0.unsqueeze(-2) - kp1_0.unsqueeze(-3)) ** 2, -1).sqrt()
    return (dist0 + dist1) / 2.0


def matches0_to_matches(matches):
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    return matches