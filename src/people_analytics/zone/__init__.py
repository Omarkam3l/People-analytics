"""Spatial zones package."""

from people_analytics.zone.engine import ZoneEngine
from people_analytics.zone.geometry import (
    compute_polygon_area,
    is_point_in_polygon,
    is_point_on_segment,
    validate_polygon_vertices,
)
from people_analytics.zone.models import Zone, ZoneMembership
from people_analytics.zone.visualization import draw_zones

__all__ = [
    "Zone",
    "ZoneEngine",
    "ZoneMembership",
    "compute_polygon_area",
    "draw_zones",
    "is_point_in_polygon",
    "is_point_on_segment",
    "validate_polygon_vertices",
]
