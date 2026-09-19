"""Trajectory builder and accumulator component."""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from people_analytics.footpoint.models import FootpointObservation
from people_analytics.trajectory.models import Trajectory


class TrajectoryBuilder:
    """Accumulates and organizes footpoint observations into time-ordered trajectories.

    Architecture & Responsibilities:
        TrajectoryBuilder (Ingestion & Normalization):
            - Accepts footpoint observations in arbitrary input order.
            - Groups observations strictly by track_id.
            - Sorts observations chronologically by frame_index upon building.
            - Detects and rejects duplicate observations for the same frame_index.

        Trajectory (Immutable Domain Entity):
            - Represents the normalized, immutable result of trajectory construction.
            - Strictly enforces single identity, chronological ordering, and duplicate constraints.
            - Preserves missing frame intervals (gaps) as unobserved gaps without synthetic points.
            - Does NOT interpolate, smooth, estimate, or fill missing frames.
    """

    def __init__(self):
        self._observations_by_track: Dict[int, List[FootpointObservation]] = defaultdict(list)

    @property
    def track_ids(self) -> List[int]:
        """List of track IDs currently registered in the builder."""
        return sorted(self._observations_by_track.keys())

    def add_observation(self, obs: FootpointObservation) -> None:
        """Add a single footpoint observation to the builder.

        Args:
            obs: FootpointObservation instance.
        """
        self._observations_by_track[obs.track_id].append(obs)

    def add_observations(self, observations: Iterable[FootpointObservation]) -> None:
        """Add a batch of footpoint observations to the builder.

        Args:
            observations: Iterable of FootpointObservation instances.
        """
        for obs in observations:
            self.add_observation(obs)

    def build_trajectory(self, track_id: int) -> Trajectory:
        """Build and return a sorted, validated Trajectory for the specified track_id.

        Args:
            track_id: ID of the track to construct.

        Returns:
            Validated Trajectory instance.

        Raises:
            KeyError: If track_id has not been observed.
            ValueError: If duplicate frame observations exist for this track.
        """
        if track_id not in self._observations_by_track:
            raise KeyError(f"Track ID {track_id} has no recorded observations in builder.")

        raw_points = self._observations_by_track[track_id]
        # Sort chronologically by frame_index
        sorted_points = sorted(raw_points, key=lambda p: p.frame_index)

        # Validation of duplicates happens inside Trajectory.__post_init__
        return Trajectory(track_id=track_id, points=tuple(sorted_points))

    def build_all(self) -> Dict[int, Trajectory]:
        """Build and return validated Trajectory instances for all observed tracks.

        Returns:
            Dictionary mapping track_id to its Trajectory.
        """
        return {tid: self.build_trajectory(tid) for tid in self.track_ids}

    def clear(self) -> None:
        """Reset builder state and clear all recorded observations."""
        self._observations_by_track.clear()
