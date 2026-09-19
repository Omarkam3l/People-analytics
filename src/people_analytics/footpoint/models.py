"""Data models for footpoint extraction."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Footpoint:
    """Raw 2D image-space contact point with sub-pixel precision."""
    x: float
    y: float

    @property
    def as_tuple(self) -> Tuple[float, float]:
        """Return coordinates as an (x, y) tuple."""
        return self.x, self.y


@dataclass(frozen=True)
class FootpointObservation:
    """Ground-contact footpoint observation associated with a tracked person identity."""
    track_id: int
    frame_index: int
    x: float
    y: float
    confidence: float = 1.0

    @property
    def point(self) -> Footpoint:
        """Return the underlying Footpoint object."""
        return Footpoint(self.x, self.y)

    @property
    def as_tuple(self) -> Tuple[float, float]:
        """Return coordinates as an (x, y) tuple."""
        return self.x, self.y
