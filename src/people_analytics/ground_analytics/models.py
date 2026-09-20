"""Data models for ground-plane movement analytics."""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from people_analytics.footpoint.models import FootpointObservation
from people_analytics.ground.models import CoordinateFrame, GroundPoint
from people_analytics.zone.geometry import validate_polygon_vertices


@dataclass(frozen=True)
class GroundObservation:
    """Immutable spatial observation on the ground plane carrying full image provenance."""
    track_id: int
    frame_index: int
    x: float
    y: float
    frame: CoordinateFrame = CoordinateFrame.ARBITRARY_PLANAR
    is_valid: bool = True
    is_extrapolated: bool = False
    source_footpoint_x: float = 0.0
    source_footpoint_y: float = 0.0
    error_code: Optional[str] = None

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError(f"track_id must be non-negative, got {self.track_id}")
        if self.frame_index < 1:
            raise ValueError(f"frame_index must be >= 1, got {self.frame_index}")
        if self.is_valid and (not math.isfinite(self.x) or not math.isfinite(self.y)):
            raise ValueError(f"Valid GroundObservation coordinates must be finite, got ({self.x}, {self.y})")

    @property
    def coordinates(self) -> Tuple[float, float]:
        """(X, Y) ground-plane coordinates."""
        return (self.x, self.y)

    @classmethod
    def from_ground_point(
        cls,
        gp: GroundPoint,
        source_footpoint: Optional[FootpointObservation] = None,
    ) -> "GroundObservation":
        """Factory creating GroundObservation from GroundPoint and source footpoint."""
        s_x = source_footpoint.x if source_footpoint is not None else 0.0
        s_y = source_footpoint.y if source_footpoint is not None else 0.0
        return cls(
            track_id=gp.track_id,
            frame_index=gp.frame_index,
            x=gp.x,
            y=gp.y,
            frame=gp.frame,
            is_valid=gp.is_valid,
            is_extrapolated=gp.is_extrapolated,
            source_footpoint_x=s_x,
            source_footpoint_y=s_y,
            error_code=gp.error_code,
        )


@dataclass(frozen=True)
class GroundTrajectory:
    """Chronological sequence of valid ground-plane observations for a single track."""
    track_id: int
    points: Tuple[GroundObservation, ...]
    coordinate_frame: CoordinateFrame = CoordinateFrame.ARBITRARY_PLANAR

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError(f"track_id must be non-negative, got {self.track_id}")
        for i, pt in enumerate(self.points):
            if pt.track_id != self.track_id:
                raise ValueError(
                    f"Mismatched track_id in trajectory: expected {self.track_id}, got {pt.track_id} at index {i}"
                )
            if not pt.is_valid:
                raise ValueError(f"GroundTrajectory cannot contain invalid GroundObservation at index {i}")
            if i > 0 and pt.frame_index <= self.points[i - 1].frame_index:
                raise ValueError(
                    f"GroundTrajectory points must be strictly monotonically increasing by frame_index: "
                    f"frame {pt.frame_index} <= previous {self.points[i - 1].frame_index}"
                )

    @property
    def start_frame(self) -> Optional[int]:
        """First observed frame index, or None if trajectory is empty."""
        return self.points[0].frame_index if self.points else None

    @property
    def end_frame(self) -> Optional[int]:
        """Last observed frame index, or None if trajectory is empty."""
        return self.points[-1].frame_index if self.points else None

    @property
    def length(self) -> int:
        """Count of observations in the trajectory."""
        return len(self.points)

    @property
    def has_gaps(self) -> bool:
        """True if any consecutive observations have missing frames between them."""
        for i in range(len(self.points) - 1):
            if self.points[i + 1].frame_index - self.points[i].frame_index > 1:
                return True
        return False

    @property
    def coordinates(self) -> List[Tuple[float, float]]:
        """List of (X, Y) coordinate tuples."""
        return [pt.coordinates for pt in self.points]

    @property
    def extrapolated_count(self) -> int:
        """Count of points outside the calibration ROI."""
        return sum(1 for pt in self.points if pt.is_extrapolated)


@dataclass(frozen=True)
class GroundHeatmapConfig:
    """Configuration for discrete planar ground heatmap accumulation."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    cell_size: float
    coordinate_frame: CoordinateFrame = CoordinateFrame.ARBITRARY_PLANAR
    allow_extrapolated: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_x) or not math.isfinite(self.max_x):
            raise ValueError(f"min_x and max_x must be finite: ({self.min_x}, {self.max_x})")
        if not math.isfinite(self.min_y) or not math.isfinite(self.max_y):
            raise ValueError(f"min_y and max_y must be finite: ({self.min_y}, {self.max_y})")
        if self.max_x <= self.min_x:
            raise ValueError(f"max_x ({self.max_x}) must be strictly greater than min_x ({self.min_x})")
        if self.max_y <= self.min_y:
            raise ValueError(f"max_y ({self.max_y}) must be strictly greater than min_y ({self.min_y})")
        if not math.isfinite(self.cell_size) or self.cell_size <= 0.0:
            raise ValueError(f"cell_size must be strictly positive and finite, got {self.cell_size}")

    @property
    def grid_width(self) -> int:
        """Number of columns in discrete grid."""
        return math.ceil((self.max_x - self.min_x) / self.cell_size)

    @property
    def grid_height(self) -> int:
        """Number of rows in discrete grid."""
        return math.ceil((self.max_y - self.min_y) / self.cell_size)

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """(rows, cols) shape of grid matrix."""
        return (self.grid_height, self.grid_width)


@dataclass(frozen=True)
class GroundHeatmapData:
    """Immutable accumulation snapshot of a planar ground heatmap."""
    raw_counts: np.ndarray
    config: GroundHeatmapConfig
    total_accumulated: int
    occupied_cells: int
    max_cell_count: float

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape (rows, cols)."""
        return self.config.grid_shape

    def to_normalized(self) -> np.ndarray:
        """Return float grid normalized to [0.0, 1.0] by maximum accumulated count."""
        if self.max_cell_count > 0.0:
            return self.raw_counts.astype(np.float64) / self.max_cell_count
        return np.zeros_like(self.raw_counts, dtype=np.float64)


@dataclass(frozen=True)
class GroundZone:
    """User-defined spatial polygon defined on the ground plane."""
    zone_id: str
    vertices: Tuple[Tuple[float, float], ...]
    name: Optional[str] = None
    coordinate_frame: CoordinateFrame = CoordinateFrame.ARBITRARY_PLANAR

    def __post_init__(self) -> None:
        if not self.zone_id or not isinstance(self.zone_id, str) or not self.zone_id.strip():
            raise ValueError(f"zone_id must be a non-empty string, got {self.zone_id!r}")

        # Validate geometry invariants and normalize with Phase 7 semantics
        validated = validate_polygon_vertices(self.vertices)
        object.__setattr__(self, "vertices", validated)


@dataclass(frozen=True)
class GroundZoneMembership:
    """Association of a GroundObservation with matched spatial GroundZone identifiers."""
    track_id: int
    frame_index: int
    x: float
    y: float
    zone_ids: Tuple[str, ...]
    coordinate_frame: CoordinateFrame = CoordinateFrame.ARBITRARY_PLANAR

    @property
    def is_in_zone(self) -> bool:
        """True if observation matched at least one ground zone."""
        return len(self.zone_ids) > 0

    @property
    def primary_zone_id(self) -> Optional[str]:
        """Matched zone ID if exactly one zone matched; None if 0 or multiple zones matched."""
        return self.zone_ids[0] if len(self.zone_ids) == 1 else None
