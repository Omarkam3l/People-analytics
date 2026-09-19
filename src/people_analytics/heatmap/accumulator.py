"""Image-space discrete grid accumulator for spatial heatmaps."""

import math
from typing import Iterable, Sequence
import numpy as np

from people_analytics.footpoint.models import FootpointObservation
from people_analytics.heatmap.models import HeatmapConfig, HeatmapData
from people_analytics.trajectory.models import Trajectory


class HeatmapAccumulator:
    """Accumulates observed footpoints onto an image-space discrete grid.

    Key Principles:
        - Pure discrete grid binning: 100% deterministic and reproducible.
        - Strict image bounds: Points outside [0, image_width) x [0, image_height) are rejected
          and increment out_of_bounds_points; they are NEVER clamped to grid edges.
        - Non-finite coordinates (NaN, +Inf, -Inf) are rejected and increment out_of_bounds_points.
        - Unobserved frames / gaps contribute zero; only valid observations contribute.
        - Default weighting is uniform (1.0 per observed footpoint).
    """

    def __init__(self, config: HeatmapConfig):
        self._config = config
        self._grid = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)
        self._total_points = 0
        self._out_of_bounds_points = 0
        self._invalid_points = 0

    @property
    def config(self) -> HeatmapConfig:
        """The configuration associated with this accumulator."""
        return self._config

    @property
    def total_points(self) -> int:
        """Total number of in-bounds points successfully accumulated."""
        return self._total_points

    @property
    def out_of_bounds_points(self) -> int:
        """Total number of out-of-frame coordinates rejected (< 0 or >= canvas)."""
        return self._out_of_bounds_points

    @property
    def invalid_points(self) -> int:
        """Total number of non-finite coordinates rejected (NaN, +Inf, -Inf)."""
        return self._invalid_points

    def add_point(self, x: float, y: float, weight: float = 1.0) -> bool:
        """Add a single image-space coordinate point.

        Args:
            x: Horizontal pixel coordinate.
            y: Vertical pixel coordinate.
            weight: Additive weight for this observation (default: 1.0).

        Returns:
            True if point was within canvas bounds and accumulated; False if rejected
            due to being out of bounds or non-finite.
        """
        # Validate finite values (reject NaN, +Inf, -Inf)
        if not math.isfinite(x) or not math.isfinite(y):
            self._invalid_points += 1
            return False

        # Validate canvas bounds: [0, image_width) x [0, image_height)
        if x < 0.0 or x >= self._config.image_width or y < 0.0 or y >= self._config.image_height:
            self._out_of_bounds_points += 1
            return False

        # Map continuous pixel coordinate to discrete grid cell
        if self._config.cell_size is not None:
            col = int(x // self._config.cell_size)
            row = int(y // self._config.cell_size)
        else:
            col = int(x * self._config.grid_width / self._config.image_width)
            row = int(y * self._config.grid_height / self._config.image_height)
            col = min(col, self._config.grid_width - 1)
            row = min(row, self._config.grid_height - 1)

        self._grid[row, col] += float(weight)
        self._total_points += 1
        return True

    def add_observation(self, obs: FootpointObservation) -> bool:
        """Add a single FootpointObservation.

        Args:
            obs: FootpointObservation instance.

        Returns:
            True if in bounds, False otherwise.
        """
        return self.add_point(obs.x, obs.y)

    def add_observations(self, observations: Iterable[FootpointObservation]) -> int:
        """Add a batch of FootpointObservation instances.

        Args:
            observations: Iterable of FootpointObservation instances.

        Returns:
            Count of points successfully accumulated.
        """
        added = 0
        for obs in observations:
            if self.add_observation(obs):
                added += 1
        return added

    def add_trajectory(self, trajectory: Trajectory) -> int:
        """Accumulate all observed footpoints from a Trajectory.

        Missing frames / gaps in the trajectory are preserved and contribute zero.

        Args:
            trajectory: Trajectory instance.

        Returns:
            Count of points successfully accumulated.
        """
        return self.add_observations(trajectory.points)

    def add_trajectories(self, trajectories: Iterable[Trajectory]) -> int:
        """Accumulate observations across multiple trajectories.

        Args:
            trajectories: Iterable of Trajectory instances.

        Returns:
            Count of points successfully accumulated.
        """
        added = 0
        for traj in trajectories:
            added += self.add_trajectory(traj)
        return added

    def build(self) -> HeatmapData:
        """Construct an immutable HeatmapData snapshot from current accumulation.

        Returns:
            HeatmapData containing raw_counts and normalized [0.0, 1.0] grids.
        """
        raw = self._grid.copy()
        max_val = float(np.max(raw)) if raw.size > 0 else 0.0

        if max_val > 0.0:
            normalized = (raw / max_val).astype(np.float32)
        else:
            normalized = np.zeros_like(raw, dtype=np.float32)

        return HeatmapData(
            raw_counts=raw,
            normalized=normalized,
            total_points=self._total_points,
            out_of_bounds_points=self._out_of_bounds_points,
            invalid_points=self._invalid_points,
            config=self._config,
        )

    def reset(self) -> None:
        """Clear all accumulated counts and reset diagnostic counters."""
        self._grid.fill(0.0)
        self._total_points = 0
        self._out_of_bounds_points = 0
        self._invalid_points = 0
