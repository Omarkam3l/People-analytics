"""Data models for ground-plane projection and planar homography."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class CoordinateFrame(str, Enum):
    """Reference coordinate system for spatial analysis."""
    IMAGE_PIXEL = "image_pixel"
    ARBITRARY_PLANAR = "arbitrary_planar"
    METRIC_GROUND = "metric_ground"


@dataclass(frozen=True)
class CalibrationPoint:
    """Pair of corresponding points in image and ground coordinates."""
    image_x: float
    image_y: float
    ground_x: float
    ground_y: float
    label: Optional[str] = None

    def __post_init__(self) -> None:
        import math
        for val, name in [
            (self.image_x, "image_x"),
            (self.image_y, "image_y"),
            (self.ground_x, "ground_x"),
            (self.ground_y, "ground_y"),
        ]:
            if not math.isfinite(val):
                raise ValueError(f"CalibrationPoint {name} must be finite, got {val}")


@dataclass(frozen=True)
class GroundPoint:
    """Ground-plane point transformed from an image-space footpoint."""
    track_id: int
    frame_index: int
    x: float
    y: float
    is_valid: bool
    is_extrapolated: bool = False
    error_code: Optional[str] = None
    frame: CoordinateFrame = CoordinateFrame.ARBITRARY_PLANAR

    @property
    def coordinates(self) -> Tuple[float, float]:
        """(x, y) coordinates on the ground plane."""
        return (self.x, self.y)


@dataclass(frozen=True)
class GroundPlaneCalibration:
    """Immutable calibration definition and homography matrix for a scene viewpoint."""
    calibration_id: str
    scene_id: str
    source_image_size: Tuple[int, int]  # (width, height)
    target_frame: CoordinateFrame
    units: str  # "pixels", "arbitrary", or "meters"
    reference_points: Tuple[CalibrationPoint, ...]
    homography_matrix: Tuple[Tuple[float, float, float], ...]  # 3x3 matrix
    inverse_homography_matrix: Tuple[Tuple[float, float, float], ...]  # 3x3 matrix
    reprojection_rmse: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.reference_points) < 4:
            raise ValueError(f"At least 4 reference points required, got {len(self.reference_points)}")
        if len(self.homography_matrix) != 3 or any(len(row) != 3 for row in self.homography_matrix):
            raise ValueError("homography_matrix must be a 3x3 matrix")
        if len(self.inverse_homography_matrix) != 3 or any(len(row) != 3 for row in self.inverse_homography_matrix):
            raise ValueError("inverse_homography_matrix must be a 3x3 matrix")
