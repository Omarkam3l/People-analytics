"""Abstract base class for multi-object tracking backends."""

from abc import ABC, abstractmethod
from typing import List, Optional

from people_analytics.detection.models import PersonDetection
from people_analytics.tracking.models import TrackerConfig, TrackObservation


class BaseTracker(ABC):
    """Abstract base tracker contract."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()

    @abstractmethod
    def update(
        self,
        detections: List[PersonDetection],
        frame_index: int,
    ) -> List[TrackObservation]:
        """Update tracker with detections from a single frame and return active observations.

        Args:
            detections: List of PersonDetection objects in the current frame.
            frame_index: 1-based frame index.

        Returns:
            List of active confirmed TrackObservation objects.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal tracker state and ID counters."""
        pass
