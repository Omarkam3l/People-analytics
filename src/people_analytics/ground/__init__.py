"""Ground-plane projection and planar homography package."""

from people_analytics.ground.homography import (
    check_collinearity,
    compute_reprojection_errors,
    create_ground_calibration,
    estimate_homography,
    normalize_points_hartley,
)
from people_analytics.ground.models import (
    CalibrationPoint,
    CoordinateFrame,
    GroundPlaneCalibration,
    GroundPoint,
)
from people_analytics.ground.projector import GroundPlaneProjector
from people_analytics.ground.visualization import draw_ground_plane_map

__all__ = [
    "CoordinateFrame",
    "CalibrationPoint",
    "GroundPoint",
    "GroundPlaneCalibration",
    "normalize_points_hartley",
    "check_collinearity",
    "estimate_homography",
    "compute_reprojection_errors",
    "create_ground_calibration",
    "GroundPlaneProjector",
    "draw_ground_plane_map",
]
