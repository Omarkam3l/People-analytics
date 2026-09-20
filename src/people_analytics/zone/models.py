"""Data models for spatial zones and zone membership."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from people_analytics.zone.geometry import is_point_in_polygon, validate_polygon_vertices


@dataclass(frozen=True)
class Zone:
    """Immutable spatial region of interest defined by a 2D polygon in image coordinates.

    Attributes:
        zone_id: Unique string identifier for the zone.
        vertices: Sequence of 2D (x, y) coordinates defining the boundary polygon.
        name: Optional descriptive human-readable label.
    """
    zone_id: str
    vertices: Tuple[Tuple[float, float], ...]
    name: Optional[str] = None

    def __post_init__(self):
        if not self.zone_id or not isinstance(self.zone_id, str):
            raise ValueError(f"zone_id must be a non-empty string, got {self.zone_id!r}")

        # Validate geometry invariants and normalize
        validated = validate_polygon_vertices(self.vertices)
        object.__setattr__(self, "vertices", validated)

    def contains_point(self, x: float, y: float, inclusive: bool = True) -> bool:
        """Test if the point (x, y) lies inside this zone.

        Semantics:
            When inclusive=True:
                - interior -> True
                - boundary edge -> True
                - vertex -> True
                - exterior -> False

            When inclusive=False:
                - interior -> True
                - boundary edge -> False
                - vertex -> False
                - exterior -> False

        Args:
            x: Horizontal image-space pixel coordinate.
            y: Vertical image-space pixel coordinate.
            inclusive: If True, points on polygon edges or vertices evaluate as inside.
                If False, points on edges or vertices evaluate as outside.

        Returns:
            True if point is inside the zone, False otherwise.
        """
        return is_point_in_polygon((x, y), self.vertices, inclusive=inclusive)


@dataclass(frozen=True)
class ZoneMembership:
    """Associates an observed footpoint with spatial zone membership.

    Attributes:
        track_id: ID of the tracked person.
        frame_index: Video frame index of the observation.
        x: Horizontal footpoint pixel coordinate.
        y: Vertical footpoint pixel coordinate.
        zone_ids: Tuple of matched zone IDs (empty if footpoint is outside all zones).
    """
    track_id: int
    frame_index: int
    x: float
    y: float
    zone_ids: Tuple[str, ...] = ()

    @property
    def is_in_zone(self) -> bool:
        """True if the footpoint lies inside at least one registered zone."""
        return len(self.zone_ids) > 0

    @property
    def primary_zone_id(self) -> Optional[str]:
        """The single matched zone ID if exactly one zone matches; otherwise None.

        Semantics:
            - zero matched zones -> None
            - exactly one matched zone -> that zone ID
            - multiple matched zones -> None

        Do not silently choose a winner among overlapping zones.
        The authoritative representation remains zone_ids.
        """
        return self.zone_ids[0] if len(self.zone_ids) == 1 else None
