"""Ground-plane dwell time and visit analytics adapter."""

from typing import Dict, Iterable, List, Optional, Sequence
from people_analytics.dwell.engine import DwellTimeEngine
from people_analytics.dwell.models import DwellConfig, ZoneVisit
from people_analytics.ground_analytics.models import (
    GroundTrajectory,
    GroundZoneMembership,
)
from people_analytics.ground_analytics.zone import GroundZoneEngine
from people_analytics.zone.models import ZoneMembership


class GroundDwellEngine:
    """Adapts GroundZoneMembership observations into temporal ZoneVisit records using Phase 8 engine."""

    def __init__(self, config: DwellConfig):
        """Initialize the ground dwell engine.

        Args:
            config: DwellConfig specifying fps and max_gap_frames tolerance.
        """
        self._config = config
        self._inner_engine = DwellTimeEngine(config)

    @property
    def config(self) -> DwellConfig:
        """Active configuration."""
        return self._config

    @property
    def track_ids(self) -> List[int]:
        """List of all registered track IDs."""
        return self._inner_engine.track_ids

    def add_membership(self, membership: GroundZoneMembership) -> None:
        """Record a single GroundZoneMembership observation.

        Args:
            membership: GroundZoneMembership instance to record.
        """
        adapted = ZoneMembership(
            track_id=membership.track_id,
            frame_index=membership.frame_index,
            x=membership.x,
            y=membership.y,
            zone_ids=membership.zone_ids,
        )
        self._inner_engine.add_membership(adapted)

    def add_memberships(self, memberships: Iterable[GroundZoneMembership]) -> int:
        """Record a collection of GroundZoneMembership observations.

        Args:
            memberships: Iterable of GroundZoneMembership instances.

        Returns:
            Count of recorded observations.
        """
        count = 0
        for m in memberships:
            self.add_membership(m)
            count += 1
        return count

    def add_trajectory(self, trajectory: GroundTrajectory, zone_engine: GroundZoneEngine) -> int:
        """Evaluate a GroundTrajectory with a GroundZoneEngine and record all resulting memberships.

        Preserves gap honesty: only actual recorded points produce memberships.

        Args:
            trajectory: GroundTrajectory instance.
            zone_engine: GroundZoneEngine configured with ground zones.

        Returns:
            Count of recorded observations.
        """
        memberships = zone_engine.evaluate_trajectory(trajectory)
        return self.add_memberships(memberships)

    def finalize_track(self, track_id: int) -> None:
        """Finalize observations for a specific track ID."""
        self._inner_engine.finalize_track(track_id)

    def finalize_all(self) -> None:
        """Finalize all currently accumulated observations into completed visits."""
        self._inner_engine.finalize_all()

    def clear(self) -> None:
        """Reset the engine state and clear all visits."""
        self._inner_engine.clear()

    @property
    def visits(self) -> List[ZoneVisit]:
        """List of all finalized ZoneVisit records in deterministic chronological order."""
        return self._inner_engine.visits

    def get_visits_for_zone(self, zone_id: str) -> List[ZoneVisit]:
        """Retrieve all visits for a specific ground zone."""
        return self._inner_engine.get_visits_for_zone(zone_id)

    def get_visits_for_track(self, track_id: int) -> List[ZoneVisit]:
        """Retrieve all visits for a specific track ID."""
        return self._inner_engine.get_visits_for_track(track_id)

    def get_total_dwell_for_zone(self, zone_id: str) -> float:
        """Total dwell time in seconds for a specific ground zone."""
        return self._inner_engine.get_total_dwell_for_zone(zone_id)

    def get_unique_visitors_for_zone(self, zone_id: str) -> int:
        """Count of distinct track IDs that visited a specific ground zone."""
        return self._inner_engine.get_unique_visitors_for_zone(zone_id)

    def get_visit_count_for_zone(self, zone_id: str) -> int:
        """Total count of visits recorded for a specific ground zone."""
        return self._inner_engine.get_visit_count_for_zone(zone_id)
