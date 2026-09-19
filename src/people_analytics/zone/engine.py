"""Zone engine managing spatial zones and evaluating footpoint memberships."""

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from people_analytics.footpoint.models import FootpointObservation
from people_analytics.trajectory.models import Trajectory
from people_analytics.zone.models import Zone, ZoneMembership


class ZoneEngine:
    """Manages spatial zones and evaluates footpoint memberships.

    Key Principles:
        - Image space only: Coordinates are evaluated directly on the image plane.
        - Multiple zones supported: Evaluates points against all registered zones independently;
          returns all matching zone IDs without arbitrary tie-breaking.
        - Gap preservation: Evaluates only actual observations in a Trajectory; missing frames
          produce zero memberships (no interpolation across unobserved intervals).
    """

    def __init__(self, zones: Optional[Sequence[Zone]] = None):
        self._zones: Dict[str, Zone] = OrderedDict()
        if zones is not None:
            for z in zones:
                self.add_zone(z)

    @property
    def zone_ids(self) -> List[str]:
        """List of registered zone identifiers in insertion order."""
        return list(self._zones.keys())

    def add_zone(self, zone: Zone) -> None:
        """Register a new spatial zone.

        Args:
            zone: Zone instance to add.

        Raises:
            ValueError: If a zone with the same zone_id already exists.
        """
        if zone.zone_id in self._zones:
            raise ValueError(f"Zone with ID {zone.zone_id!r} is already registered.")
        self._zones[zone.zone_id] = zone

    def get_zone(self, zone_id: str) -> Zone:
        """Retrieve a registered zone by its identifier.

        Args:
            zone_id: Identifier of the zone.

        Returns:
            The registered Zone object.

        Raises:
            KeyError: If zone_id is not registered.
        """
        if zone_id not in self._zones:
            raise KeyError(f"Zone {zone_id!r} not found in ZoneEngine.")
        return self._zones[zone_id]

    def remove_zone(self, zone_id: str) -> None:
        """Remove a registered zone by its identifier."""
        if zone_id in self._zones:
            del self._zones[zone_id]

    def clear(self) -> None:
        """Clear all registered zones."""
        self._zones.clear()

    def evaluate_point(self, x: float, y: float, inclusive: bool = True) -> Tuple[str, ...]:
        """Test a 2D coordinate against all registered zones.

        Args:
            x: Horizontal pixel coordinate.
            y: Vertical pixel coordinate.
            inclusive: If True, edge and vertex points evaluate as inside.

        Returns:
            Tuple of matching zone_ids in registration order (empty if outside all zones).
        """
        matched = []
        for zone_id, zone in self._zones.items():
            if zone.contains_point(x, y, inclusive=inclusive):
                matched.append(zone_id)
        return tuple(matched)

    def evaluate_observation(
        self,
        obs: FootpointObservation,
        inclusive: bool = True,
    ) -> ZoneMembership:
        """Evaluate zone membership for a single FootpointObservation.

        Args:
            obs: FootpointObservation instance.
            inclusive: Closed-boundary inclusive flag.

        Returns:
            ZoneMembership capturing the observation and all matching zone_ids.
        """
        matched_ids = self.evaluate_point(obs.x, obs.y, inclusive=inclusive)
        return ZoneMembership(
            track_id=obs.track_id,
            frame_index=obs.frame_index,
            x=obs.x,
            y=obs.y,
            zone_ids=matched_ids,
        )

    def evaluate_trajectory(
        self,
        trajectory: Trajectory,
        inclusive: bool = True,
    ) -> List[ZoneMembership]:
        """Evaluate zone membership for all observed points in a Trajectory.

        Gap Semantics:
            Observations exist only for observed frames. Missing frames/gaps
            produce ZERO memberships. Membership is never interpolated across gaps.

        Args:
            trajectory: Trajectory instance.
            inclusive: Closed-boundary inclusive flag.

        Returns:
            List of ZoneMembership objects, one per observed footpoint.
        """
        return [self.evaluate_observation(pt, inclusive=inclusive) for pt in trajectory.points]
