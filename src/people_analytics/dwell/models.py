"""Data models and configuration for dwell time and visit analytics."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


@dataclass(frozen=True)
class DwellConfig:
    """Configuration for dwell time and visit analysis.

    Attributes:
        fps: Video sampling rate in frames per second (must be strictly positive).
        max_gap_frames: Maximum number of unobserved consecutive frames tolerated within
            a single visit to the same zone (default: 0).
            - Definition: missing_frames = current_frame_index - last_observed_frame_index - 1.
            - When missing_frames <= max_gap_frames: visit survives the gap.
            - When missing_frames > max_gap_frames: visit terminates at last_observed_frame_index.
            - 0: Strict observation policy. Only strictly contiguous frames (missing_frames == 0)
              continue the visit. Any missing frame terminates the visit.
            - K > 0: Bounded gap tolerance policy. A visit remains open across up to K missing
              frames if the target reappears in the same zone.
    """
    fps: float
    max_gap_frames: int = 0

    def __post_init__(self):
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"fps must be a strictly positive finite number: {self.fps}")
        if self.max_gap_frames < 0:
            raise ValueError(f"max_gap_frames must be non-negative: {self.max_gap_frames}")


@dataclass(frozen=True)
class ZoneVisit:
    """Immutable record of an observed person visit to a spatial zone.

    Time & Duration Semantics:
        - Reuses trajectory timestamp convention: t = (frame_index - 1) / fps.
        - entry_frame: First frame index where the target is observed inside the zone.
        - last_observed_frame: Final frame index where the target is observed inside the zone.
        - duration_seconds = exit_timestamp - entry_timestamp = (last_observed_frame - entry_frame) / fps.
        - Single-frame visit (entry_frame == last_observed_frame):
            - duration_seconds = 0.0s (an isolated point observation cannot establish an elapsed continuous span).
            - observation_count = 1 (discrete contact count is preserved).
        - Both duration_seconds and observation_count are exposed for complete analytical transparency.

    Bounded Gap Interpretation:
        - When max_gap_frames > 0 and a visit survives an observation gap, duration_seconds represents
          a gap-tolerant observed interval.
        - This does NOT imply the person's position was observed continuously.
        - Missing frames are NEVER added to observation_count.
        - No spatial coordinates or synthetic observations are ever interpolated.

    Invariants:
        - entry_frame <= last_observed_frame
        - entry_timestamp <= exit_timestamp
        - observation_count >= 1
        - observation_count <= last_observed_frame - entry_frame + 1
    """
    track_id: int
    zone_id: str
    entry_frame: int
    last_observed_frame: int
    entry_timestamp: float
    exit_timestamp: float
    duration_seconds: float
    observation_count: int

    def __post_init__(self):
        if not self.zone_id or not isinstance(self.zone_id, str):
            raise ValueError(f"zone_id must be a non-empty string, got {self.zone_id!r}")
        if self.entry_frame <= 0:
            raise ValueError(f"entry_frame must be >= 1, got {self.entry_frame}")
        if self.last_observed_frame < self.entry_frame:
            raise ValueError(
                f"last_observed_frame ({self.last_observed_frame}) cannot precede "
                f"entry_frame ({self.entry_frame})"
            )
        if self.entry_timestamp < 0.0 or self.exit_timestamp < self.entry_timestamp:
            raise ValueError(
                f"Invalid timestamps: entry={self.entry_timestamp}, exit={self.exit_timestamp}"
            )
        if self.observation_count < 1:
            raise ValueError(f"observation_count must be >= 1, got {self.observation_count}")
        max_possible_obs = self.last_observed_frame - self.entry_frame + 1
        if self.observation_count > max_possible_obs:
            raise ValueError(
                f"observation_count ({self.observation_count}) exceeds frame span ({max_possible_obs})"
            )


@dataclass(frozen=True)
class ZoneAnalytics:
    """Aggregate visit and dwell statistics for a spatial zone.

    Attributes:
        zone_id: Identifier of the analyzed zone.
        total_visits: Total number of discrete visits recorded.
        unique_visitors: Count of distinct track IDs that visited the zone.
        total_dwell_time: Total elapsed dwell time across all visits in seconds.
        average_dwell_time: Mean visit duration in seconds (total_dwell_time / total_visits).
        max_dwell_time: Maximum single visit duration in seconds.
        total_observations: Sum of all observed in-zone frames across all visits.
    """
    zone_id: str
    total_visits: int
    unique_visitors: int
    total_dwell_time: float
    average_dwell_time: float
    max_dwell_time: float
    total_observations: int


@dataclass(frozen=True)
class PersonVisitSummary:
    """Visit history and aggregate dwell time for a single tracked person.

    Attributes:
        track_id: Identity of the tracked target.
        visits: Chronological sequence of completed ZoneVisit records for this target.
        total_dwell_time: Sum of dwell time across all zones visited by this target.
        zones_visited: Lexicographically sorted tuple of unique zone IDs visited.
    """
    track_id: int
    visits: Tuple[ZoneVisit, ...]
    total_dwell_time: float
    zones_visited: Tuple[str, ...]
