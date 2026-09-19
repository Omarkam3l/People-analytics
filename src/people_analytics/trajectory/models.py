"""Data models for trajectory representation."""

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

from people_analytics.footpoint.models import FootpointObservation


@dataclass(frozen=True)
class Trajectory:
    """Time-ordered sequence of observed footpoints for a single tracked identity.

    Invariants:
        - All observations belong to the same track_id.
        - Observations are ordered strictly chronologically by frame_index.
        - No duplicate frame_index is permitted.
        - Missing frame intervals are preserved as unobserved gaps without interpolation.
    """
    track_id: int
    points: Tuple[FootpointObservation, ...] = ()

    def __post_init__(self):
        last_frame: Optional[int] = None
        for i, pt in enumerate(self.points):
            if pt.track_id != self.track_id:
                raise ValueError(
                    f"Observation at index {i} has track_id {pt.track_id}, "
                    f"expected {self.track_id}"
                )
            if last_frame is not None:
                if pt.frame_index == last_frame:
                    raise ValueError(
                        f"Duplicate observation for track_id {self.track_id} at frame {pt.frame_index}"
                    )
                if pt.frame_index < last_frame:
                    raise ValueError(
                        f"Observations must be strictly chronological: frame {pt.frame_index} "
                        f"appears after {last_frame}"
                    )
            last_frame = pt.frame_index

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[FootpointObservation]:
        return iter(self.points)

    def __getitem__(self, index: int) -> FootpointObservation:
        return self.points[index]

    @property
    def is_empty(self) -> bool:
        """True if the trajectory contains no observations."""
        return len(self.points) == 0

    @property
    def start_frame(self) -> Optional[int]:
        """First observed frame index, or None if empty."""
        return self.points[0].frame_index if self.points else None

    @property
    def end_frame(self) -> Optional[int]:
        """Last observed frame index, or None if empty."""
        return self.points[-1].frame_index if self.points else None

    @property
    def frame_indices(self) -> Tuple[int, ...]:
        """Tuple of all observed frame indices in chronological order."""
        return tuple(p.frame_index for p in self.points)

    @property
    def coordinates(self) -> Tuple[Tuple[float, float], ...]:
        """Tuple of all observed (x, y) coordinates in chronological order."""
        return tuple(p.as_tuple for p in self.points)

    @property
    def has_gaps(self) -> bool:
        """Indicates whether any temporal gaps exist within the observed trajectory.

        Semantics:
            Returns True if and only if there exists any adjacent pair of observations where:
                frame_indices[i + 1] - frame_indices[i] > 1

        Examples:
            - frame_indices = [10, 11, 14] -> has_gaps = True  (gap at frames 12-13)
            - frame_indices = [10, 11, 12] -> has_gaps = False (strictly contiguous)
            - frame_indices = [5]          -> has_gaps = False (single observation)
            - frame_indices = []           -> has_gaps = False (empty trajectory)

        Note:
            has_gaps is a pure data-level property describing missing frame observations.
            It is completely independent of downstream visualization connection or rendering policies.
        """
        if len(self.points) < 2:
            return False
        return any(
            self.points[i + 1].frame_index - self.points[i].frame_index > 1
            for i in range(len(self.points) - 1)
        )

    def get_point_at_frame(self, frame_index: int) -> Optional[FootpointObservation]:
        """Retrieve the exact spatial observation recorded at a specific frame index.

        Exact Lookup Semantics:
            - If an observation was recorded for this frame_index, returns that FootpointObservation.
            - If no observation was recorded for this frame_index (i.e. frame was missed, lost, or out of range),
              returns None.

        CRITICAL:
            This method performs an EXACT discrete lookup. It does NOT return the nearest observation,
            does NOT interpolate missing coordinates, and does NOT estimate or extrapolate spatial positions.

        Args:
            frame_index: The integer frame index to query.

        Returns:
            The FootpointObservation at that frame, or None.
        """
        # Binary search for efficiency across chronological points
        low = 0
        high = len(self.points) - 1
        while low <= high:
            mid = (low + high) // 2
            mid_frame = self.points[mid].frame_index
            if mid_frame == frame_index:
                return self.points[mid]
            elif mid_frame < frame_index:
                low = mid + 1
            else:
                high = mid - 1
        return None

    @staticmethod
    def get_timestamp(frame_index: int, fps: float) -> float:
        """Derive elapsed sequence seconds from a 1-indexed MOT frame index: t = (frame_index - 1) / fps.

        Args:
            frame_index: 1-indexed video frame number (Frame 1 = 0.0s).
            fps: Video frames per second (must be > 0).

        Returns:
            Timestamp in seconds.

        Raises:
            ValueError: If fps <= 0.
        """
        if fps <= 0:
            raise ValueError(f"FPS must be strictly positive: {fps}")
        return (frame_index - 1) / float(fps)
