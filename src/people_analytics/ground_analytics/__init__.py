"""Ground-plane movement analytics package."""

from people_analytics.ground_analytics.comparison import (
    GroundComparisonReport,
    compare_image_and_ground_analytics,
)
from people_analytics.ground_analytics.dwell import GroundDwellEngine
from people_analytics.ground_analytics.heatmap import GroundHeatmapAccumulator
from people_analytics.ground_analytics.models import (
    GroundHeatmapConfig,
    GroundHeatmapData,
    GroundObservation,
    GroundTrajectory,
    GroundZone,
    GroundZoneMembership,
)
from people_analytics.ground_analytics.trajectory import GroundTrajectoryBuilder
from people_analytics.ground_analytics.visualization import (
    draw_dual_view_diagnostics,
    draw_ground_analytics_map,
)
from people_analytics.ground_analytics.zone import (
    GroundZoneEngine,
    point_in_ground_polygon,
)

__all__ = [
    "GroundObservation",
    "GroundTrajectory",
    "GroundHeatmapConfig",
    "GroundHeatmapData",
    "GroundZone",
    "GroundZoneMembership",
    "GroundTrajectoryBuilder",
    "GroundHeatmapAccumulator",
    "point_in_ground_polygon",
    "GroundZoneEngine",
    "GroundDwellEngine",
    "GroundComparisonReport",
    "compare_image_and_ground_analytics",
    "draw_ground_analytics_map",
    "draw_dual_view_diagnostics",
]
