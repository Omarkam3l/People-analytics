"""Data models and configuration for spatial heatmaps."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class HeatmapConfig:
    """Configuration for spatial heatmap accumulation grid.

    Grid Dimension Derivation:
        - If cell_size is specified, grid dimensions are determined by a deterministic ceiling policy:
            grid_width = ceil(image_width / cell_size)
            grid_height = ceil(image_height / cell_size)
          When image dimensions are not evenly divisible by cell_size, the final column
          (col = grid_width - 1) and final row (row = grid_height - 1) serve as partial boundary
          cells covering the remaining fractional pixel interval, ensuring all valid canvas coordinates
          [0, image_width) x [0, image_height) map into a valid cell.
        - If cell_size is None, explicit grid_width and grid_height are used (defaulting to
          full 1-to-1 pixel canvas dimensions).

    Attributes:
        image_width: Width of the camera image canvas in pixels.
        image_height: Height of the camera image canvas in pixels.
        cell_size: Uniform square cell size in pixels for grid downsampling (optional).
        grid_width: Explicit width of the accumulation grid (optional).
        grid_height: Explicit height of the accumulation grid (optional).
    """
    image_width: int
    image_height: int
    cell_size: Optional[int] = None
    grid_width: Optional[int] = None
    grid_height: Optional[int] = None

    def __post_init__(self):
        if self.image_width <= 0:
            raise ValueError(f"image_width must be strictly positive: {self.image_width}")
        if self.image_height <= 0:
            raise ValueError(f"image_height must be strictly positive: {self.image_height}")

        # Determine grid dimensions
        if self.cell_size is not None:
            if self.cell_size <= 0:
                raise ValueError(f"cell_size must be strictly positive: {self.cell_size}")
            calculated_w = math.ceil(self.image_width / self.cell_size)
            calculated_h = math.ceil(self.image_height / self.cell_size)
            object.__setattr__(self, "grid_width", calculated_w)
            object.__setattr__(self, "grid_height", calculated_h)
        else:
            final_w = self.grid_width if self.grid_width is not None else self.image_width
            final_h = self.grid_height if self.grid_height is not None else self.image_height

            if final_w <= 0:
                raise ValueError(f"grid_width must be strictly positive: {final_w}")
            if final_h <= 0:
                raise ValueError(f"grid_height must be strictly positive: {final_h}")

            object.__setattr__(self, "grid_width", final_w)
            object.__setattr__(self, "grid_height", final_h)


@dataclass(frozen=True)
class HeatmapData:
    """Immutable snapshot of accumulated image-space spatial density.

    Terminology & Interpretation:
        - raw_counts: Number of observed footpoints accumulated per discrete grid cell.
        - normalized: Relative visitation intensity scaled to [0.0, 1.0] by dividing by max(raw_counts).
        - Domain: This is an image-space visitation/count heatmap.
          It is NOT a probability density function.
          It is NOT physical or metric density (no physical area or meter units).
          It is NOT ground-plane occupancy.

    Attributes:
        raw_counts: 2D numpy array of shape (grid_height, grid_width) with raw hit counts.
        normalized: 2D numpy array of shape (grid_height, grid_width) normalized to [0.0, 1.0].
        total_points: Total number of valid observed points accumulated.
        out_of_bounds_points: Total number of rejected out-of-frame coordinates (< 0 or >= canvas).
        invalid_points: Total number of rejected non-finite coordinates (NaN, +Inf, -Inf).
        config: HeatmapConfig used for the accumulation grid.
    """
    raw_counts: np.ndarray
    normalized: np.ndarray
    total_points: int
    out_of_bounds_points: int
    invalid_points: int
    config: HeatmapConfig

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape as (grid_height, grid_width)."""
        return int(self.config.grid_height), int(self.config.grid_width)

    @property
    def max_count(self) -> float:
        """Maximum raw count observed in any grid cell."""
        return float(np.max(self.raw_counts)) if self.raw_counts.size > 0 else 0.0
