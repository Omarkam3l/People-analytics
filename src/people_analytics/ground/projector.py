"""Projector transforming image-space footpoints to ground-plane coordinates."""

import math
from typing import List, Optional, Sequence, Tuple
import numpy as np

from people_analytics.footpoint.models import FootpointObservation
from people_analytics.ground.models import GroundPlaneCalibration, GroundPoint
from people_analytics.trajectory.models import Trajectory


def _point_in_convex_polygon(px: float, py: float, vertices: Sequence[Tuple[float, float]]) -> bool:
    """Check if point (px, py) lies inside a convex polygon using cross products."""
    n = len(vertices)
    if n < 3:
        return False

    sign = None
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        current_sign = cross > 0
        if sign is None:
            sign = current_sign
        elif sign != current_sign:
            return False
    return True


class GroundPlaneProjector:
    """Transforms image coordinates into ground-plane coordinates via calibrated homography."""

    def __init__(self, calibration: GroundPlaneCalibration):
        """Initialize the projector with an immutable GroundPlaneCalibration.

        Args:
            calibration: Calibrated GroundPlaneCalibration instance.
        """
        self._calib = calibration
        self._H = np.array(calibration.homography_matrix, dtype=np.float64)
        self._H_inv = np.array(calibration.inverse_homography_matrix, dtype=np.float64)

        # Polygon of reference image points for extrapolation checking
        self._ref_image_polygon = [(p.image_x, p.image_y) for p in calibration.reference_points]

    @property
    def calibration(self) -> GroundPlaneCalibration:
        """Active calibration configuration."""
        return self._calib

    def project_point(
        self,
        image_x: float,
        image_y: float,
        track_id: int = 0,
        frame_index: int = 0,
    ) -> GroundPoint:
        """Project a single image coordinate pair to ground plane.

        Args:
            image_x: Pixel X coordinate in image raster.
            image_y: Pixel Y coordinate in image raster.
            track_id: Identifier of the tracked person (default: 0).
            frame_index: Video frame index (default: 0).

        Returns:
            GroundPoint with validity, extrapolation, and error flags.
        """
        if not math.isfinite(image_x) or not math.isfinite(image_y):
            return GroundPoint(
                track_id=track_id,
                frame_index=frame_index,
                x=float("nan"),
                y=float("nan"),
                is_valid=False,
                is_extrapolated=False,
                error_code="NON_FINITE_INPUT",
                frame=self._calib.target_frame,
            )

        # Homogeneous projection: [x', y', w']^T = H @ [x, y, 1]^T
        p_img = np.array([image_x, image_y, 1.0], dtype=np.float64)
        p_ground = self._H @ p_img

        w_prime = float(p_ground[2])
        if not math.isfinite(w_prime) or w_prime <= 1e-7:
            return GroundPoint(
                track_id=track_id,
                frame_index=frame_index,
                x=float("nan"),
                y=float("nan"),
                is_valid=False,
                is_extrapolated=False,
                error_code="PROJECTIVE_DENOMINATOR_TOO_SMALL",
                frame=self._calib.target_frame,
            )

        X_ground = float(p_ground[0] / w_prime)
        Y_ground = float(p_ground[1] / w_prime)

        if not math.isfinite(X_ground) or not math.isfinite(Y_ground):
            return GroundPoint(
                track_id=track_id,
                frame_index=frame_index,
                x=X_ground,
                y=Y_ground,
                is_valid=False,
                is_extrapolated=False,
                error_code="NON_FINITE_OUTPUT",
                frame=self._calib.target_frame,
            )

        # Check extrapolation relative to calibration reference polygon
        is_inside = _point_in_convex_polygon(image_x, image_y, self._ref_image_polygon)
        is_extrapolated = not is_inside

        return GroundPoint(
            track_id=track_id,
            frame_index=frame_index,
            x=X_ground,
            y=Y_ground,
            is_valid=True,
            is_extrapolated=is_extrapolated,
            error_code=None,
            frame=self._calib.target_frame,
        )

    def project_observation(self, observation: FootpointObservation) -> GroundPoint:
        """Project a FootpointObservation into a GroundPoint.

        Args:
            observation: FootpointObservation from Phase 4.

        Returns:
            GroundPoint preserving track_id and frame_index.
        """
        return self.project_point(
            image_x=observation.x,
            image_y=observation.y,
            track_id=observation.track_id,
            frame_index=observation.frame_index,
        )

    def project_batch(self, observations: Sequence[FootpointObservation]) -> List[GroundPoint]:
        """Project a batch of FootpointObservation instances.

        Args:
            observations: Sequence of FootpointObservation instances.

        Returns:
            List of GroundPoint records in corresponding order.
        """
        return [self.project_observation(obs) for obs in observations]

    def project_trajectory(self, trajectory: Trajectory) -> List[GroundPoint]:
        """Project all footpoint observations in a Trajectory onto the ground plane.

        Missing frames/gaps are preserved: only actual recorded points are projected.

        Args:
            trajectory: Trajectory instance from Phase 5.

        Returns:
            List of GroundPoint instances in chronological order.
        """
        return [self.project_observation(pt) for pt in trajectory.points]

    def project_ground_to_image(self, ground_x: float, ground_y: float) -> Tuple[float, float]:
        """Back-project a ground-plane coordinate into image pixel coordinates via H^-1.

        Args:
            ground_x: Ground X coordinate.
            ground_y: Ground Y coordinate.

        Returns:
            (image_x, image_y) pixel coordinates.

        Raises:
            ValueError: If input is non-finite or back-projects behind camera / to horizon.
        """
        if not math.isfinite(ground_x) or not math.isfinite(ground_y):
            raise ValueError(f"Ground coordinates must be finite, got ({ground_x}, {ground_y})")

        p_ground = np.array([ground_x, ground_y, 1.0], dtype=np.float64)
        p_img = self._H_inv @ p_ground

        w_img = float(p_img[2])
        if not math.isfinite(w_img) or abs(w_img) <= 1e-9:
            raise ValueError(f"Ground point ({ground_x}, {ground_y}) back-projects to singular image coordinate.")

        img_x = float(p_img[0] / w_img)
        img_y = float(p_img[1] / w_img)

        return (img_x, img_y)
