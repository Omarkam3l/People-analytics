"""Planar homography estimation, validation, and conditioning diagnostics."""

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from people_analytics.ground.models import (
    CalibrationPoint,
    CoordinateFrame,
    GroundPlaneCalibration,
)


def check_collinearity(points: np.ndarray, area_threshold: float = 1e-4) -> None:
    """Check whether any triplet among the points is collinear.

    Args:
        points: (N, 2) array of coordinates.
        area_threshold: Minimum triangle area threshold for collinearity test.

    Raises:
        ValueError: If fewer than 3 points or any 3 points are collinear.
    """
    n = len(points)
    if n < 3:
        raise ValueError(f"At least 3 points required for collinearity check, got {n}")

    for i, j, k in itertools.combinations(range(n), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        # Cross product of vectors (p2 - p1) and (p3 - p1) equals 2 * area of triangle
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        area = 0.5 * abs(cross)
        if area < area_threshold:
            raise ValueError(
                f"Points at indices ({i}, {j}, {k}) are collinear (triangle area {area:.6e} < {area_threshold:.6e})."
            )


def normalize_points_hartley(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley isotropic normalization: translate centroid to origin and scale average distance to sqrt(2).

    Args:
        points: (N, 2) array of coordinates.

    Returns:
        Tuple of (T, normalized_points):
            - T: (3, 3) similarity transformation matrix.
            - normalized_points: (N, 2) normalized coordinate array.

    Raises:
        ValueError: If points are degenerate or have zero spread.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) points, got shape {pts.shape}")

    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))

    if mean_dist < 1e-9:
        raise ValueError("Cannot normalize points with zero or near-zero spatial spread.")

    scale = math.sqrt(2.0) / mean_dist

    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Transform points: [x, y, 1] @ T.T
    homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    norm_homo = homo @ T.T
    normalized_pts = norm_homo[:, :2] / norm_homo[:, 2:3]

    return T, normalized_pts


def estimate_homography(
    src_points: Sequence[Tuple[float, float]],
    dst_points: Sequence[Tuple[float, float]],
    check_condition: bool = True,
) -> np.ndarray:
    """Estimate a 3x3 planar homography H mapping src_points to dst_points using Normalized DLT.

    p_dst ~ H @ p_src

    Args:
        src_points: Sequence of N >= 4 (x, y) coordinates in source space (e.g. image pixels).
        dst_points: Sequence of N >= 4 (x, y) coordinates in destination space (e.g. ground plane).
        check_condition: If True, validates non-collinearity and condition number.

    Returns:
        (3, 3) float64 homography matrix H normalized so H[2, 2] == 1.0 (or unit Frobenius norm).

    Raises:
        ValueError: If point counts mismatch, N < 4, points are collinear, or H is degenerate.
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)

    if len(src) != len(dst):
        raise ValueError(f"Mismatched point counts: src has {len(src)}, dst has {len(dst)}")
    if len(src) < 4:
        raise ValueError(f"Homography estimation requires at least 4 point correspondences, got {len(src)}")

    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise ValueError("All calibration points must have finite coordinates.")

    if check_condition:
        check_collinearity(src)
        check_collinearity(dst)

    # 1. Hartley Normalization
    T_src, norm_src = normalize_points_hartley(src)
    T_dst, norm_dst = normalize_points_hartley(dst)

    # 2. Build 2N x 9 matrix A
    n = len(src)
    A = np.zeros((2 * n, 9), dtype=np.float64)

    for i in range(n):
        x, y = norm_src[i]
        X, Y = norm_dst[i]

        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, x * X, y * X, X]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, x * Y, y * Y, Y]

    # 3. Singular Value Decomposition
    _, _, vt = np.linalg.svd(A)
    h_norm = vt[-1].reshape(3, 3)

    # 4. De-normalization: H = T_dst^-1 @ H_norm @ T_src
    H = np.linalg.inv(T_dst) @ h_norm @ T_src

    # 5. Scale normalization
    if abs(H[2, 2]) > 1e-9:
        H = H / H[2, 2]
    else:
        norm = np.linalg.norm(H)
        if norm > 1e-9:
            H = H / norm

    # 6. Degeneracy and condition checks
    if check_condition:
        det = abs(np.linalg.det(H))
        if det < 1e-9:
            raise ValueError(f"Estimated homography is near-singular (determinant {det:.6e} < 1e-9).")

        svd_s = np.linalg.svd(H, compute_uv=False)
        cond = svd_s[0] / svd_s[-1] if svd_s[-1] > 0 else float("inf")
        if cond > 1e7:
            raise ValueError(f"Estimated homography is ill-conditioned (condition number {cond:.2e} > 1e7).")

    return H


def compute_reprojection_errors(
    H: np.ndarray,
    src_points: Sequence[Tuple[float, float]],
    dst_points: Sequence[Tuple[float, float]],
) -> Dict[str, float]:
    """Compute forward and backward reprojection errors across correspondences.

    Args:
        H: (3, 3) homography matrix mapping src to dst.
        src_points: Sequence of source (x, y) points.
        dst_points: Sequence of destination (X, Y) points.

    Returns:
        Dict with keys:
            - 'mean_forward': mean Euclidean forward error.
            - 'median_forward': median Euclidean forward error.
            - 'rmse_forward': root-mean-squared forward error.
            - 'max_forward': maximum Euclidean forward error.
            - 'mean_backward': mean Euclidean backward error.
            - 'rmse_backward': root-mean-squared backward error.
            - 'max_backward': maximum Euclidean backward error.
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    n = len(src)

    # Forward transform: src -> dst
    src_h = np.hstack([src, np.ones((n, 1), dtype=np.float64)])  # (N, 3)
    proj_dst_h = (H @ src_h.T).T  # (N, 3)

    valid_fwd = abs(proj_dst_h[:, 2]) > 1e-9
    if not np.all(valid_fwd):
        raise ValueError("Some reference points projected to zero or invalid homogeneous w.")

    proj_dst = proj_dst_h[:, :2] / proj_dst_h[:, 2:3]
    fwd_diffs = np.linalg.norm(proj_dst - dst, axis=1)

    # Backward transform: dst -> src
    H_inv = np.linalg.inv(H)
    dst_h = np.hstack([dst, np.ones((n, 1), dtype=np.float64)])
    proj_src_h = (H_inv @ dst_h.T).T

    valid_bwd = abs(proj_src_h[:, 2]) > 1e-9
    if not np.all(valid_bwd):
        raise ValueError("Some reference points back-projected to zero or invalid homogeneous w.")

    proj_src = proj_src_h[:, :2] / proj_src_h[:, 2:3]
    bwd_diffs = np.linalg.norm(proj_src - src, axis=1)

    return {
        "mean_forward": float(np.mean(fwd_diffs)),
        "median_forward": float(np.median(fwd_diffs)),
        "rmse_forward": float(np.sqrt(np.mean(fwd_diffs ** 2))),
        "max_forward": float(np.max(fwd_diffs)),
        "mean_backward": float(np.mean(bwd_diffs)),
        "rmse_backward": float(np.sqrt(np.mean(bwd_diffs ** 2))),
        "max_backward": float(np.max(bwd_diffs)),
    }


def create_ground_calibration(
    calibration_id: str,
    scene_id: str,
    source_image_size: Tuple[int, int],
    target_frame: CoordinateFrame,
    units: str,
    reference_points: Sequence[CalibrationPoint],
    metadata: Optional[Dict[str, Any]] = None,
) -> GroundPlaneCalibration:
    """Factory creating an immutable GroundPlaneCalibration with estimated H and diagnostics.

    Args:
        calibration_id: Unique identifier for this calibration profile.
        scene_id: Identifier of the static camera scene/viewpoint.
        source_image_size: (width, height) of image canvas.
        target_frame: Target coordinate frame (e.g. METRIC_GROUND or ARBITRARY_PLANAR).
        units: Explicit unit string ('meters', 'pixels', 'arbitrary').
        reference_points: Sequence of at least 4 CalibrationPoint correspondences.
        metadata: Optional dictionary with custom parameters or survey notes.

    Returns:
        GroundPlaneCalibration instance with computed homography matrices and RMSE.
    """
    if target_frame == CoordinateFrame.METRIC_GROUND and units.lower() != "meters":
        raise ValueError(
            f"target_frame is METRIC_GROUND but units is '{units}'; metric frame requires units='meters'."
        )

    src_pts = [(p.image_x, p.image_y) for p in reference_points]
    dst_pts = [(p.ground_x, p.ground_y) for p in reference_points]

    H = estimate_homography(src_pts, dst_pts, check_condition=True)
    H_inv = np.linalg.inv(H)
    if abs(H_inv[2, 2]) > 1e-9:
        H_inv = H_inv / H_inv[2, 2]

    errors = compute_reprojection_errors(H, src_pts, dst_pts)

    h_tuple = tuple(tuple(float(v) for v in row) for row in H)
    h_inv_tuple = tuple(tuple(float(v) for v in row) for row in H_inv)

    return GroundPlaneCalibration(
        calibration_id=calibration_id,
        scene_id=scene_id,
        source_image_size=source_image_size,
        target_frame=target_frame,
        units=units,
        reference_points=tuple(reference_points),
        homography_matrix=h_tuple,
        inverse_homography_matrix=h_inv_tuple,
        reprojection_rmse=errors["rmse_forward"],
        metadata=dict(metadata or {}),
    )
