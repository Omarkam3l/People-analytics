"""Footpoint extraction components."""

from abc import ABC, abstractmethod
import math
from typing import List, Tuple

from people_analytics.footpoint.models import Footpoint, FootpointObservation
from people_analytics.tracking.models import TrackObservation


class BaseFootpointExtractor(ABC):
    """Abstract base class for extracting ground contact points from bounding boxes."""

    @abstractmethod
    def extract_point(self, bbox_xyxy: Tuple[float, float, float, float]) -> Footpoint:
        """Extract a footpoint from a raw (x1, y1, x2, y2) bounding box.

        Args:
            bbox_xyxy: (x1, y1, x2, y2) bounding box coordinates.

        Returns:
            Footpoint object in image coordinates.

        Raises:
            ValueError: If bounding box coordinates are non-finite or degenerate.
        """
        pass

    def extract_from_observation(self, obs: TrackObservation) -> FootpointObservation:
        """Extract a FootpointObservation from a confirmed TrackObservation.

        Args:
            obs: TrackObservation from a tracker.

        Returns:
            FootpointObservation containing track identity, frame index, and footpoint.
        """
        pt = self.extract_point(obs.bbox_xyxy)
        return FootpointObservation(
            track_id=obs.track_id,
            frame_index=obs.frame_index,
            x=pt.x,
            y=pt.y,
            confidence=obs.confidence,
        )

    def extract_batch(self, observations: List[TrackObservation]) -> List[FootpointObservation]:
        """Extract footpoint observations for a batch of track observations.

        Args:
            observations: List of TrackObservation objects.

        Returns:
            List of FootpointObservation objects corresponding to the input observations.
        """
        return [self.extract_from_observation(obs) for obs in observations]


class BottomCenterFootpointExtractor(BaseFootpointExtractor):
    """Extracts footpoint as the bottom-center of the bounding box.

    Mathematical formulation:
        x_foot = (x1 + x2) / 2
        y_foot = y2

    Coordinate Conventions:
        - Image space, origin (0, 0) at top-left corner.
        - +X rightward, +Y downward.
        - No silent clipping: points outside image boundaries are preserved without distortion.
    """

    def extract_point(self, bbox_xyxy: Tuple[float, float, float, float]) -> Footpoint:
        """Extract the bottom-center point of a bounding box.

        Args:
            bbox_xyxy: Tuple of (x1, y1, x2, y2) coordinates.

        Returns:
            Footpoint with (x_foot, y_foot).

        Raises:
            ValueError: If coordinates are non-finite, or width/height <= 0.
        """
        x1, y1, x2, y2 = bbox_xyxy

        # Validate finite values
        for val, name in [(x1, "x1"), (y1, "y1"), (x2, "x2"), (y2, "y2")]:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Bounding box coordinate '{name}' is not finite: {val}")

        width = x2 - x1
        height = y2 - y1

        if width <= 0:
            raise ValueError(f"Invalid bounding box width: {width:.4f} (must be strictly positive, x2 > x1)")
        if height <= 0:
            raise ValueError(f"Invalid bounding box height: {height:.4f} (must be strictly positive, y2 > y1)")

        x_foot = (x1 + x2) / 2.0
        y_foot = float(y2)

        return Footpoint(x=x_foot, y=y_foot)
