"""Ground-plane trajectory construction and temporal grouping."""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from people_analytics.ground.models import CoordinateFrame
from people_analytics.ground_analytics.models import GroundObservation, GroundTrajectory


class GroundTrajectoryBuilder:
    """Stateful accumulator building chronological GroundTrajectory records per track."""

    def __init__(
        self,
        coordinate_frame: CoordinateFrame = CoordinateFrame.ARBITRARY_PLANAR,
        include_extrapolated: Optional[bool] = None,
        allow_extrapolated: bool = True,
    ):
        extrap = include_extrapolated if include_extrapolated is not None else allow_extrapolated
        self._frame = coordinate_frame
        self._allow_extrapolated = extrap
        self._observations_by_track: Dict[int, List[GroundObservation]] = defaultdict(list)
        self._invalid_observations: List[GroundObservation] = []
        self._invalid_error_codes: Dict[str, int] = defaultdict(int)
        self._extrapolated_excluded_count: int = 0

    @property
    def coordinate_frame(self) -> CoordinateFrame:
        """Active coordinate frame."""
        return self._frame

    @property
    def allow_extrapolated(self) -> bool:
        """Default policy: whether extrapolated points are included in trajectories (default: True)."""
        return self._allow_extrapolated

    @property
    def include_extrapolated(self) -> bool:
        """Alias for allow_extrapolated (default: True)."""
        return self._allow_extrapolated

    @property
    def invalid_count(self) -> int:
        """Count of invalid GroundObservation instances safely excluded from trajectories."""
        return len(self._invalid_observations)

    @property
    def invalid_ignored_count(self) -> int:
        """Alias for invalid_count."""
        return len(self._invalid_observations)

    @property
    def invalid_error_codes(self) -> Dict[str, int]:
        """Breakdown of invalid projections by error code."""
        return dict(self._invalid_error_codes)

    @property
    def invalid_observations(self) -> Tuple[GroundObservation, ...]:
        """Diagnostic access to dropped invalid observations."""
        return tuple(self._invalid_observations)

    @property
    def extrapolated_excluded_count(self) -> int:
        """Count of valid extrapolated observations excluded by policy."""
        return self._extrapolated_excluded_count

    @property
    def track_ids(self) -> List[int]:
        """List of all registered track IDs."""
        return sorted(self._observations_by_track.keys())

    def add_observation(self, observation: GroundObservation) -> bool:
        """Add a single GroundObservation to its corresponding track sequence.

        Invalid projections (is_valid == False) are safely excluded and recorded for diagnostics.
        Extrapolated points (is_extrapolated == True) are governed by include_extrapolated.

        Args:
            observation: GroundObservation to record.

        Returns:
            True if the observation was valid and added; False if ignored/excluded.
        """
        if not observation.is_valid:
            self._invalid_observations.append(observation)
            code = observation.error_code or "UNKNOWN"
            self._invalid_error_codes[code] += 1
            return False

        if not self._allow_extrapolated and observation.is_extrapolated:
            self._extrapolated_excluded_count += 1
            return False

        self._observations_by_track[observation.track_id].append(observation)
        return True

    def add_observations(self, observations: Iterable[GroundObservation]) -> int:
        """Add a collection of GroundObservation instances.

        Args:
            observations: Iterable of GroundObservation instances.

        Returns:
            Count of valid observations accepted.
        """
        accepted = 0
        for obs in observations:
            if self.add_observation(obs):
                accepted += 1
        return accepted

    def build(self, track_id: int) -> GroundTrajectory:
        """Construct a GroundTrajectory for a specific track ID.

        Points are strictly sorted chronologically by frame_index.

        Args:
            track_id: Identifier of track to build.

        Returns:
            GroundTrajectory instance.

        Raises:
            KeyError: If track_id has no recorded observations.
            ValueError: If duplicate observations exist for the same frame_index.
        """
        if track_id not in self._observations_by_track:
            raise KeyError(f"No observations recorded for track_id {track_id}")

        raw_points = self._observations_by_track[track_id]
        sorted_points = sorted(raw_points, key=lambda p: p.frame_index)

        # Check for duplicate frames
        for i in range(len(sorted_points) - 1):
            if sorted_points[i + 1].frame_index == sorted_points[i].frame_index:
                raise ValueError(
                    f"Duplicate GroundObservation found for track {track_id} at frame {sorted_points[i].frame_index}"
                )

        return GroundTrajectory(
            track_id=track_id,
            points=tuple(sorted_points),
            coordinate_frame=self._frame,
        )

    def build_all(self) -> Dict[int, GroundTrajectory]:
        """Construct GroundTrajectory records for all registered tracks in ascending track_id order.

        Returns:
            Dict mapping track_id to GroundTrajectory.
        """
        result: Dict[int, GroundTrajectory] = {}
        for tid in sorted(self._observations_by_track.keys()):
            result[tid] = self.build(tid)
        return result

    def clear(self) -> None:
        """Reset internal accumulator and clear all recorded observations and diagnostics."""
        self._observations_by_track.clear()
        self._invalid_observations.clear()
        self._invalid_error_codes.clear()
        self._extrapolated_excluded_count = 0
