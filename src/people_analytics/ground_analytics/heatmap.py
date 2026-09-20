"""Planar ground-space heatmap accumulation and spatial density."""

import math
from typing import Iterable, List, Optional, Tuple
import numpy as np

from people_analytics.ground_analytics.models import (
    GroundHeatmapConfig,
    GroundHeatmapData,
    GroundObservation,
    GroundTrajectory,
)


class GroundHeatmapAccumulator:
    """Discrete 2D grid accumulator for continuous planar ground coordinates."""

    def __init__(self, config: GroundHeatmapConfig):
        self._config = config
        self._grid = np.zeros(config.grid_shape, dtype=np.float64)
        self._total_points: int = 0
        self._out_of_bounds_points: int = 0
        self._invalid_points: int = 0
        self._invalid_error_codes: dict[str, int] = {}
        self._extrapolated_rejected_points: int = 0

    @property
    def config(self) -> GroundHeatmapConfig:
        """Active configuration."""
        return self._config

    @property
    def total_accumulated(self) -> int:
        """Count of observations successfully accumulated into the grid."""
        return self._total_points

    @property
    def out_of_bounds_points(self) -> int:
        """Count of valid points rejected because they fell outside [min_x, max_x] x [min_y, max_y]."""
        return self._out_of_bounds_points

    @property
    def invalid_points(self) -> int:
        """Count of points rejected due to invalid projection or non-finite coordinates."""
        return self._invalid_points

    @property
    def invalid_error_codes(self) -> dict[str, int]:
        """Breakdown of rejected invalid points by error code."""
        return dict(self._invalid_error_codes)

    @property
    def extrapolated_rejected_points(self) -> int:
        """Count of points rejected because allow_extrapolated was False."""
        return self._extrapolated_rejected_points

    def to_normalized(self) -> np.ndarray:
        """Return float grid normalized to [0.0, 1.0] by maximum accumulated count."""
        max_val = float(np.max(self._grid)) if self._grid.size > 0 else 0.0
        if max_val > 0.0:
            return self._grid.astype(np.float64) / max_val
        return np.zeros_like(self._grid, dtype=np.float64)

    def add_observation(self, observation: GroundObservation, weight: float = 1.0) -> bool:
        """Accumulate a single GroundObservation into the planar grid.

        Coordinate mapping:
            col = floor((X - min_x) / cell_size)
            row = floor((Y - min_y) / cell_size)
        Boundary inclusive semantics:
            points at max_x or max_y map to the final grid column or row.
        Rejections:
            points outside bounding box are rejected without clamping.

        Args:
            observation: GroundObservation to accumulate.
            weight: Additive weight for this observation (default: 1.0).

        Returns:
            True if accumulated; False if rejected (OOB, invalid, or extrapolated).
        """
        if not observation.is_valid:
            self._invalid_points += 1
            code = observation.error_code or "UNKNOWN"
            self._invalid_error_codes[code] = self._invalid_error_codes.get(code, 0) + 1
            return False

        x, y = observation.x, observation.y
        if not math.isfinite(x) or not math.isfinite(y):
            self._invalid_points += 1
            self._invalid_error_codes["NON_FINITE_COORDINATE"] = self._invalid_error_codes.get("NON_FINITE_COORDINATE", 0) + 1
            return False

        if not self._config.allow_extrapolated and observation.is_extrapolated:
            self._extrapolated_rejected_points += 1
            return False

        # Check planar bounding box
        if x < self._config.min_x or x > self._config.max_x or y < self._config.min_y or y > self._config.max_y:
            self._out_of_bounds_points += 1
            return False

        # Map to grid indices
        cols = self._config.grid_width
        rows = self._config.grid_height

        if math.isclose(x, self._config.max_x, abs_tol=1e-9):
            col = cols - 1
        else:
            col = int(math.floor((x - self._config.min_x) / self._config.cell_size))

        if math.isclose(y, self._config.max_y, abs_tol=1e-9):
            row = rows - 1
        else:
            row = int(math.floor((y - self._config.min_y) / self._config.cell_size))

        col = min(max(col, 0), cols - 1)
        row = min(max(row, 0), rows - 1)

        self._grid[row, col] += weight
        self._total_points += 1
        return True

    def add_observations(self, observations: Iterable[GroundObservation], weight: float = 1.0) -> int:
        """Accumulate an iterable of GroundObservation instances.

        Args:
            observations: Iterable of GroundObservation instances.
            weight: Additive weight per observation (default: 1.0).

        Returns:
            Count of observations successfully accumulated.
        """
        count = 0
        for obs in observations:
            if self.add_observation(obs, weight=weight):
                count += 1
        return count

    def add_trajectory(self, trajectory: GroundTrajectory, weight: float = 1.0) -> int:
        """Accumulate all observations in a GroundTrajectory.

        Preserves gap honesty: only actual recorded observations are accumulated.

        Args:
            trajectory: GroundTrajectory instance.
            weight: Additive weight per point (default: 1.0).

        Returns:
            Count of points successfully accumulated.
        """
        return self.add_observations(trajectory.points, weight=weight)

    def build(self) -> GroundHeatmapData:
        """Construct an immutable GroundHeatmapData snapshot of the accumulated grid.

        Returns:
            GroundHeatmapData instance.
        """
        occupied = int(np.count_nonzero(self._grid))
        max_val = float(np.max(self._grid)) if self._grid.size > 0 else 0.0

        return GroundHeatmapData(
            raw_counts=self._grid.copy(),
            config=self._config,
            total_accumulated=self._total_points,
            occupied_cells=occupied,
            max_cell_count=max_val,
        )

    def reset(self) -> None:
        """Reset the accumulator and clear all accumulated values."""
        self._grid.fill(0.0)
        self._total_points = 0
        self._out_of_bounds_points = 0
        self._invalid_points = 0
        self._invalid_error_codes.clear()
        self._extrapolated_rejected_points = 0
