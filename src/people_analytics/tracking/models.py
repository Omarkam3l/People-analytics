"""Data models and configuration for multi-object tracking."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class TrackState(Enum):
    """Lifecycle states of a tracked person."""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    TERMINATED = "terminated"


@dataclass(frozen=True)
class TrackObservation:
    """Represents a single observation of a tracked person at a specific frame."""
    track_id: int
    frame_index: int
    bb_left: float
    bb_top: float
    bb_width: float
    bb_height: float
    confidence: float
    state: TrackState = TrackState.CONFIRMED

    @property
    def bbox_xywh(self) -> Tuple[float, float, float, float]:
        """Bounding box in (left, top, width, height) format."""
        return self.bb_left, self.bb_top, self.bb_width, self.bb_height

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        """Bounding box in (x1, y1, x2, y2) format."""
        return self.bb_left, self.bb_top, self.bb_left + self.bb_width, self.bb_top + self.bb_height


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration options for ByteTrack multi-object tracker."""
    high_threshold: float = 0.5     # Threshold for stage-1 high-confidence matching
    low_threshold: float = 0.1      # Threshold for stage-2 low-confidence recovery
    match_threshold: float = 0.5    # Minimum IoU for association
    min_hits: int = 2               # Consecutive frames to confirm a tentative track
    max_lost_frames: int = 30       # Maximum frames to hold a lost track before termination
