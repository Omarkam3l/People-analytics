"""Spatial ground-zone evaluation and point-in-polygon membership."""

from collections import OrderedDict
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from people_analytics.ground_analytics.models import (
    GroundObservation,
    GroundTrajectory,
    GroundZone,
    GroundZoneMembership,
)
from people_analytics.zone.geometry import is_point_in_polygon


def point_in_ground_polygon(
    px: float,
    py: float,
    vertices: Sequence[Tuple[float, float]],
    inclusive: bool = True,
) -> bool:
    """Determine if point (px, py) lies inside a ground-plane polygon.

    Uses Phase 7 geometric algorithms ensuring exact mathematical parity for
    boundary, vertex, and interior inclusion.

    Args:
        px: X coordinate.
        py: Y coordinate.
        vertices: Sequence of polygon (X, Y) vertices.
        inclusive: If True, points on boundary edges or vertices evaluate to True.

    Returns:
        True if the point lies inside or on the boundary (when inclusive=True).
    """
    if not math.isfinite(px) or not math.isfinite(py):
        return False
    return is_point_in_polygon((px, py), vertices, inclusive=inclusive)


class GroundZoneEngine:
    """Manages ground-plane spatial zones and evaluates observations for zone membership."""

    def __init__(
        self,
        zones: Sequence[GroundZone],
        allow_extrapolated: bool = True,
        include_extrapolated: Optional[bool] = None,
    ):
        """Initialize with a collection of GroundZone definitions.

        Args:
            zones: Sequence of GroundZone instances.
            allow_extrapolated: Default policy governing whether extrapolated observations
                are eligible for zone membership (default: True).
            include_extrapolated: Alias for allow_extrapolated.

        Raises:
            ValueError: If duplicate zone_ids are present.
        """
        extrap = include_extrapolated if include_extrapolated is not None else allow_extrapolated
        self._allow_extrapolated = extrap
        self._zones: Dict[str, GroundZone] = OrderedDict()
        for z in sorted(zones, key=lambda z: z.zone_id):
            if z.zone_id in self._zones:
                raise ValueError(f"Duplicate GroundZone ID detected: '{z.zone_id}'")
            self._zones[z.zone_id] = z

    @property
    def allow_extrapolated(self) -> bool:
        """Default policy: whether extrapolated points are included in zone evaluations (default: True)."""
        return self._allow_extrapolated

    @property
    def include_extrapolated(self) -> bool:
        """Alias for allow_extrapolated (default: True)."""
        return self._allow_extrapolated

    @property
    def zones(self) -> List[GroundZone]:
        """List of registered GroundZone definitions in deterministic order."""
        return list(self._zones.values())

    @property
    def zone_ids(self) -> List[str]:
        """Sorted list of registered zone IDs."""
        return list(self._zones.keys())

    def get_zone(self, zone_id: str) -> GroundZone:
        """Retrieve a registered GroundZone by its identifier."""
        if zone_id not in self._zones:
            raise KeyError(f"Zone '{zone_id}' not found in GroundZoneEngine.")
        return self._zones[zone_id]

    def evaluate_point_ids(
        self,
        x: float,
        y: float,
        inclusive: bool = True,
    ) -> Tuple[str, ...]:
        """Test a 2D ground coordinate against all registered zones.

        Returns:
            Tuple of matching zone_ids sorted lexicographically (empty if outside all zones).
            This guarantees deterministic ordering matching Phase 7 semantics.
        """
        matched: List[str] = []
        for zid, zone in self._zones.items():
            if point_in_ground_polygon(x, y, zone.vertices, inclusive=inclusive):
                matched.append(zid)
        return tuple(sorted(matched))

    def evaluate_point(
        self,
        x: float,
        y: float,
        track_id: int = 0,
        frame_index: int = 0,
        inclusive: bool = True,
    ) -> GroundZoneMembership:
        """Evaluate a single coordinate point against all registered ground zones.

        Args:
            x: Ground X coordinate.
            y: Ground Y coordinate.
            track_id: Identifier of the tracked person.
            frame_index: Video frame index.
            inclusive: If True, boundary edges count as inside.

        Returns:
            GroundZoneMembership record with matched zone_ids.
        """
        matched = self.evaluate_point_ids(x, y, inclusive=inclusive)
        return GroundZoneMembership(
            track_id=track_id,
            frame_index=frame_index,
            x=x,
            y=y,
            zone_ids=matched,
        )

    def evaluate_observation(
        self,
        observation: GroundObservation,
        inclusive: bool = True,
    ) -> GroundZoneMembership:
        """Evaluate a GroundObservation for spatial zone membership.

        Invalid projections are safely evaluated to zero matching zones.
        Extrapolated points are evaluated to zero matching zones if allow_extrapolated is False.

        Args:
            observation: GroundObservation to evaluate.
            inclusive: Boundary inclusion policy.

        Returns:
            GroundZoneMembership record.
        """
        if not observation.is_valid:
            return GroundZoneMembership(
                track_id=observation.track_id,
                frame_index=observation.frame_index,
                x=observation.x,
                y=observation.y,
                zone_ids=(),
                coordinate_frame=observation.frame,
            )

        if not self._allow_extrapolated and observation.is_extrapolated:
            return GroundZoneMembership(
                track_id=observation.track_id,
                frame_index=observation.frame_index,
                x=observation.x,
                y=observation.y,
                zone_ids=(),
                coordinate_frame=observation.frame,
            )

        return self.evaluate_point(
            x=observation.x,
            y=observation.y,
            track_id=observation.track_id,
            frame_index=observation.frame_index,
            inclusive=inclusive,
        )

    def evaluate_trajectory(
        self,
        trajectory: GroundTrajectory,
        inclusive: bool = True,
    ) -> List[GroundZoneMembership]:
        """Evaluate all observations in a GroundTrajectory.

        Preserves gaps: only actual recorded points produce membership records.

        Args:
            trajectory: GroundTrajectory instance.
            inclusive: Boundary inclusion policy.

        Returns:
            List of GroundZoneMembership records in chronological order.
        """
        return [self.evaluate_observation(pt, inclusive=inclusive) for pt in trajectory.points]
