"""Dwell time and visit analytics engine."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from people_analytics.dwell.models import (
    DwellConfig,
    PersonVisitSummary,
    ZoneAnalytics,
    ZoneVisit,
)
from people_analytics.trajectory.models import Trajectory
from people_analytics.zone.engine import ZoneEngine
from people_analytics.zone.models import ZoneMembership


@dataclass
class _ActiveVisit:
    """Internal mutable tracking state for an in-progress zone visit."""
    track_id: int
    zone_id: str
    entry_frame: int
    last_observed_frame: int
    observation_count: int


def _create_visit(active: _ActiveVisit, fps: float) -> ZoneVisit:
    """Convert an active visit into an immutable finalized ZoneVisit."""
    entry_ts = (active.entry_frame - 1) / float(fps)
    exit_ts = (active.last_observed_frame - 1) / float(fps)
    duration = exit_ts - entry_ts

    return ZoneVisit(
        track_id=active.track_id,
        zone_id=active.zone_id,
        entry_frame=active.entry_frame,
        last_observed_frame=active.last_observed_frame,
        entry_timestamp=entry_ts,
        exit_timestamp=exit_ts,
        duration_seconds=duration,
        observation_count=active.observation_count,
    )


class DwellTimeEngine:
    """Accumulates ZoneMembership observations and constructs finalized ZoneVisit records.

    Architecture & Invariants:
        - Ingestion: Accepts ZoneMembership observations in streaming or batch mode.
        - Chronological Ordering: Groups observations per track and sorts chronologically
          by frame_index. Duplicate observations for the same (track_id, frame_index) raise ValueError.
        - Deterministic Semantics:
            - Independent Zone Tracking: Overlapping zones are tracked independently per (track_id, zone_id)
              using the authoritative zone_ids tuple (no dependence on primary_zone_id).
            - Gap Definition:
                missing_frames = current_frame - last_observed_frame - 1
                - If missing_frames <= max_gap_frames: visit survives the gap.
                - If missing_frames > max_gap_frames: visit terminates at last_observed_frame.
            - Duration: duration_seconds = (last_observed_frame - entry_frame) / fps.
            - Single-frame contacts: duration_seconds = 0.0s, observation_count = 1.
            - Finalization: ZoneMembership does not carry upstream TrackState. Active visits are
              sealed at their last_observed_frame upon observed zone exit, gap threshold exceedance,
              or end-of-input / trajectory completion.
    """

    def __init__(self, config: DwellConfig):
        self._config = config
        self._memberships_by_track: Dict[int, List[ZoneMembership]] = defaultdict(list)
        self._finalized_visits: List[ZoneVisit] = []
        self._dirty: bool = False

    @property
    def config(self) -> DwellConfig:
        """Active configuration for this dwell engine."""
        return self._config

    @property
    def track_ids(self) -> List[int]:
        """List of all tracked target IDs registered in the engine."""
        return sorted(self._memberships_by_track.keys())

    def add_membership(self, membership: ZoneMembership) -> None:
        """Add a single ZoneMembership observation to the engine.

        Args:
            membership: ZoneMembership instance to record.
        """
        self._memberships_by_track[membership.track_id].append(membership)
        self._dirty = True

    def add_memberships(self, memberships: Iterable[ZoneMembership]) -> int:
        """Add a collection of ZoneMembership observations.

        Args:
            memberships: Iterable of ZoneMembership instances.

        Returns:
            Count of added observations.
        """
        count = 0
        for m in memberships:
            self.add_membership(m)
            count += 1
        return count

    def add_trajectory(self, trajectory: Trajectory, zone_engine: ZoneEngine) -> int:
        """Evaluate a Trajectory using a ZoneEngine and record all resulting memberships.

        Missing frames/gaps in the Trajectory are naturally preserved: only actual
        recorded FootpointObservation points produce ZoneMembership records.

        Args:
            trajectory: Trajectory instance to evaluate.
            zone_engine: ZoneEngine configured with spatial zones.

        Returns:
            Count of added observations.
        """
        memberships = zone_engine.evaluate_trajectory(trajectory)
        return self.add_memberships(memberships)

    def finalize_track(self, track_id: int) -> None:
        """Ensure all observations for a specific track are finalized."""
        self._ensure_built()

    def finalize_all(self) -> None:
        """Explicitly finalize all currently accumulated observations into completed visits.

        Note:
            Finalization is also executed lazily and idempotently upon accessing `.visits`
            or any analytics query method. Calling finalize_all() explicitly is optional.
        """
        self._ensure_built()

    def clear(self) -> None:
        """Reset the engine state and remove all recorded observations and visits."""
        self._memberships_by_track.clear()
        self._finalized_visits.clear()
        self._dirty = False

    @property
    def visits(self) -> List[ZoneVisit]:
        """List of all finalized ZoneVisit records in deterministic chronological order.

        Lazy Finalization & Idempotency:
            Accessing this property lazily compiles and finalizes all accumulated
            observations into visit records if new observations have been added
            since the last build (equivalent to calling finalize_all()).

            This operation is completely deterministic, idempotent, and non-destructive:
            underlying observations are preserved, repeated queries return identical
            results without duplicating visits, and a defensive copy of the visit
            list is returned.
        """
        self._ensure_built()
        return list(self._finalized_visits)

    def get_visits_for_zone(self, zone_id: str) -> List[ZoneVisit]:
        """Retrieve all finalized visits for a specific spatial zone.

        Args:
            zone_id: Identifier of the zone.

        Returns:
            List of ZoneVisit records in chronological order.
        """
        return [v for v in self.visits if v.zone_id == zone_id]

    def get_visits_for_track(self, track_id: int) -> List[ZoneVisit]:
        """Retrieve all finalized visits for a specific tracked person.

        Args:
            track_id: Identifier of the tracked person.

        Returns:
            List of ZoneVisit records in chronological order.
        """
        return [v for v in self.visits if v.track_id == track_id]

    def compute_zone_analytics(self, zone_id: str) -> ZoneAnalytics:
        """Compute aggregate statistics for a specific spatial zone.

        Args:
            zone_id: Identifier of the zone.

        Returns:
            ZoneAnalytics snapshot.
        """
        zone_visits = self.get_visits_for_zone(zone_id)
        total_visits = len(zone_visits)
        unique_visitors = len(set(v.track_id for v in zone_visits))
        total_dwell = sum(v.duration_seconds for v in zone_visits)
        avg_dwell = total_dwell / total_visits if total_visits > 0 else 0.0
        max_dwell = max((v.duration_seconds for v in zone_visits), default=0.0)
        total_obs = sum(v.observation_count for v in zone_visits)

        return ZoneAnalytics(
            zone_id=zone_id,
            total_visits=total_visits,
            unique_visitors=unique_visitors,
            total_dwell_time=total_dwell,
            average_dwell_time=avg_dwell,
            max_dwell_time=max_dwell,
            total_observations=total_obs,
        )

    def compute_all_zone_analytics(
        self,
        zone_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, ZoneAnalytics]:
        """Compute aggregate statistics across multiple spatial zones.

        Args:
            zone_ids: Optional collection of zone IDs to analyze. If None,
                analyzes all zones present in recorded visits.

        Returns:
            Dictionary mapping zone_id to its ZoneAnalytics summary.
        """
        if zone_ids is not None:
            target_zones = sorted(zone_ids)
        else:
            target_zones = sorted(set(v.zone_id for v in self.visits))

        return {zid: self.compute_zone_analytics(zid) for zid in target_zones}

    def compute_person_summary(self, track_id: int) -> PersonVisitSummary:
        """Compute visit history and aggregate dwell time for a tracked person.

        Args:
            track_id: Target person identifier.

        Returns:
            PersonVisitSummary snapshot.
        """
        person_visits = self.get_visits_for_track(track_id)
        total_dwell = sum(v.duration_seconds for v in person_visits)
        zones_visited = tuple(sorted(set(v.zone_id for v in person_visits)))

        return PersonVisitSummary(
            track_id=track_id,
            visits=tuple(person_visits),
            total_dwell_time=total_dwell,
            zones_visited=zones_visited,
        )

    def _ensure_built(self) -> None:
        """Internal build step to segment observations into deterministic visits."""
        if not self._dirty:
            return

        finalized: List[ZoneVisit] = []

        # Process each track in sorted track_id order
        for track_id in sorted(self._memberships_by_track.keys()):
            raw_obs = self._memberships_by_track[track_id]
            if not raw_obs:
                continue

            # Sort observations chronologically by frame_index
            sorted_obs = sorted(raw_obs, key=lambda o: o.frame_index)

            # Validate duplicate frame_index
            last_frame: Optional[int] = None
            for o in sorted_obs:
                if last_frame is not None and o.frame_index == last_frame:
                    raise ValueError(
                        f"Duplicate observation for track_id {track_id} at frame {o.frame_index}"
                    )
                last_frame = o.frame_index

            # Active visits state machine per zone
            active_visits: Dict[str, _ActiveVisit] = {}

            for obs in sorted_obs:
                current_zones: Set[str] = set(obs.zone_ids)

                # 1. Finalize active visits for zones no longer occupied
                for active_zone in list(active_visits.keys()):
                    if active_zone not in current_zones:
                        active = active_visits.pop(active_zone)
                        finalized.append(_create_visit(active, self._config.fps))

                # 2. Process currently occupied zones
                for zone_id in current_zones:
                    if zone_id in active_visits:
                        active = active_visits[zone_id]
                        gap = obs.frame_index - active.last_observed_frame - 1

                        if gap <= self._config.max_gap_frames:
                            # Visit continues within acceptable gap
                            active.last_observed_frame = obs.frame_index
                            active.observation_count += 1
                        else:
                            # Gap exceeded tolerance: close previous visit and open new one
                            finalized.append(_create_visit(active, self._config.fps))
                            active_visits[zone_id] = _ActiveVisit(
                                track_id=track_id,
                                zone_id=zone_id,
                                entry_frame=obs.frame_index,
                                last_observed_frame=obs.frame_index,
                                observation_count=1,
                            )
                    else:
                        # New visit starts
                        active_visits[zone_id] = _ActiveVisit(
                            track_id=track_id,
                            zone_id=zone_id,
                            entry_frame=obs.frame_index,
                            last_observed_frame=obs.frame_index,
                            observation_count=1,
                        )

            # 3. Finalize all remaining open visits at end-of-input for this track
            for active in active_visits.values():
                finalized.append(_create_visit(active, self._config.fps))

        # Sort all finalized visits deterministically: (entry_frame, track_id, zone_id)
        finalized.sort(key=lambda v: (v.entry_frame, v.track_id, v.zone_id))
        self._finalized_visits = finalized
        self._dirty = False
