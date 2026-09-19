"""Unit and integration tests for image-space heatmap layer."""

import numpy as np
import pytest

from people_analytics.footpoint.models import FootpointObservation
from people_analytics.heatmap import (
    HeatmapAccumulator,
    HeatmapConfig,
    HeatmapData,
    overlay_heatmap,
    render_heatmap,
)
from people_analytics.trajectory.models import Trajectory


# ====================================================================
# Configuration & Model Tests
# ====================================================================

def test_heatmap_config_defaults():
    config = HeatmapConfig(image_width=1920, image_height=1080)
    assert config.image_width == 1920
    assert config.image_height == 1080
    assert config.grid_width == 1920
    assert config.grid_height == 1080


def test_heatmap_config_cell_size():
    config = HeatmapConfig(image_width=1920, image_height=1080, cell_size=10)
    assert config.cell_size == 10
    assert config.grid_width == 192
    assert config.grid_height == 108


def test_heatmap_config_non_divisible_cell_size_ceiling():
    """Verify ceiling policy when image dimensions are not divisible by cell_size:
    1920 / 50 = 38.4 -> 39
    1080 / 50 = 21.6 -> 22
    """
    config = HeatmapConfig(image_width=1920, image_height=1080, cell_size=50)
    assert config.grid_width == 39
    assert config.grid_height == 22

    acc = HeatmapAccumulator(config)
    # Test point in the final partial cell
    assert acc.add_point(1919.0, 1079.0) is True
    data = acc.build()
    assert data.shape == (22, 39)
    assert data.raw_counts[21, 38] == 1.0


def test_heatmap_config_explicit_grid():
    config = HeatmapConfig(image_width=1920, image_height=1080, grid_width=100, grid_height=50)
    assert config.grid_width == 100
    assert config.grid_height == 50


def test_heatmap_config_invalid_dimensions():
    with pytest.raises(ValueError, match="image_width must be strictly positive"):
        HeatmapConfig(image_width=0, image_height=1080)

    with pytest.raises(ValueError, match="image_height must be strictly positive"):
        HeatmapConfig(image_width=1920, image_height=-10)

    with pytest.raises(ValueError, match="cell_size must be strictly positive"):
        HeatmapConfig(image_width=1920, image_height=1080, cell_size=0)


# ====================================================================
# Accumulation & Determinism Tests
# ====================================================================

def test_empty_accumulation():
    config = HeatmapConfig(image_width=100, image_height=100, cell_size=10)
    acc = HeatmapAccumulator(config)
    data = acc.build()

    assert data.total_points == 0
    assert data.out_of_bounds_points == 0
    assert data.invalid_points == 0
    assert data.max_count == 0.0
    assert data.shape == (10, 10)
    assert np.all(data.raw_counts == 0.0)
    assert np.all(data.normalized == 0.0)


def test_single_point_accumulation():
    config = HeatmapConfig(image_width=100, image_height=100, cell_size=10)
    acc = HeatmapAccumulator(config)

    # Point at (25, 45) -> col = 2, row = 4
    success = acc.add_point(25.0, 45.0)
    assert success is True

    data = acc.build()
    assert data.total_points == 1
    assert data.out_of_bounds_points == 0
    assert data.max_count == 1.0
    assert data.raw_counts[4, 2] == 1.0
    assert data.normalized[4, 2] == 1.0
    # Other cells must be zero
    assert np.sum(data.raw_counts) == 1.0


def test_repeated_point_accumulation():
    config = HeatmapConfig(image_width=100, image_height=100, cell_size=10)
    acc = HeatmapAccumulator(config)

    for _ in range(5):
        acc.add_point(15.0, 15.0)

    data = acc.build()
    assert data.total_points == 5
    assert data.max_count == 5.0
    assert data.raw_counts[1, 1] == 5.0
    assert data.normalized[1, 1] == 1.0


def test_multiple_distinct_points():
    config = HeatmapConfig(image_width=100, image_height=100, cell_size=10)
    acc = HeatmapAccumulator(config)

    acc.add_point(15.0, 15.0)  # col 1, row 1
    acc.add_point(15.0, 15.0)  # col 1, row 1 (total 2)
    acc.add_point(85.0, 85.0)  # col 8, row 8 (total 1)

    data = acc.build()
    assert data.total_points == 3
    assert data.max_count == 2.0
    assert data.raw_counts[1, 1] == 2.0
    assert data.raw_counts[8, 8] == 1.0
    assert data.normalized[1, 1] == 1.0
    assert data.normalized[8, 8] == 0.5


# ====================================================================
# Boundary & Out-of-Bounds Rejection Tests
# ====================================================================

def test_boundary_points_inclusive():
    config = HeatmapConfig(image_width=100, image_height=100, cell_size=10)
    acc = HeatmapAccumulator(config)

    # Top-left corner (0, 0)
    assert acc.add_point(0.0, 0.0) is True
    # Bottom-right corner just inside canvas
    assert acc.add_point(99.9, 99.9) is True

    data = acc.build()
    assert data.raw_counts[0, 0] == 1.0
    assert data.raw_counts[9, 9] == 1.0


def test_out_of_bounds_points_rejected_without_clamping():
    """Verify points outside frame canvas are rejected and not clamped to edges."""
    config = HeatmapConfig(image_width=100, image_height=100, cell_size=10)
    acc = HeatmapAccumulator(config)

    # Negative coordinates
    assert acc.add_point(-1.0, 50.0) is False
    assert acc.add_point(50.0, -10.0) is False

    # Coordinates at or beyond width/height
    assert acc.add_point(100.0, 50.0) is False   # x == width is out of [0, width)
    assert acc.add_point(50.0, 100.0) is False   # y == height is out of [0, height)
    assert acc.add_point(150.0, 200.0) is False

    data = acc.build()
    assert data.total_points == 0
    assert data.out_of_bounds_points == 5
    # Grid must remain completely zero (no false boundary spikes)
    assert np.all(data.raw_counts == 0.0)


def test_non_finite_coordinates_rejected():
    """Verify NaN, +Inf, -Inf coordinates are tracked in invalid_points and do not corrupt accumulator."""
    config = HeatmapConfig(image_width=100, image_height=100, cell_size=10)
    acc = HeatmapAccumulator(config)

    assert acc.add_point(float("nan"), 50.0) is False
    assert acc.add_point(50.0, float("nan")) is False
    assert acc.add_point(float("inf"), 50.0) is False
    assert acc.add_point(50.0, float("-inf")) is False
    assert acc.add_point(float("nan"), float("nan")) is False

    data = acc.build()
    assert data.total_points == 0
    assert data.invalid_points == 5
    assert data.out_of_bounds_points == 0
    assert np.all(data.raw_counts == 0.0)


def test_cell_size_boundary_mapping_distinguishes_rescaled_model():
    """Verify exact floor(x / cell_size) mapping around boundary thresholds:
    Image: 1920x1080, cell_size: 50 -> grid: 39x22
    - x = 49.9 -> col 0 (rescaled model would yield 49.9 * 39 / 1920 = 1.013 -> col 1)
    - x = 50.0 -> col 1
    - x = 1899.9 -> col 37
    - x = 1900.0 -> col 38 (start of partial 20px cell)
    - x = 1919.9 -> col 38 (inside partial 20px cell)
    Equivalent Y transitions:
    - y = 49.9 -> row 0
    - y = 50.0 -> row 1
    - y = 1049.9 -> row 20
    - y = 1050.0 -> row 21 (start of partial 30px cell)
    - y = 1079.9 -> row 21 (inside partial 30px cell)
    """
    config = HeatmapConfig(image_width=1920, image_height=1080, cell_size=50)
    assert config.grid_width == 39
    assert config.grid_height == 22

    acc = HeatmapAccumulator(config)

    # Test X boundaries with constant y = 0
    acc.add_point(49.9, 0.0)      # col 0, row 0
    acc.add_point(50.0, 0.0)      # col 1, row 0
    acc.add_point(1899.9, 0.0)    # col 37, row 0
    acc.add_point(1900.0, 0.0)    # col 38, row 0 (partial cell)
    acc.add_point(1919.9, 0.0)    # col 38, row 0 (partial cell)

    # Test Y boundaries with constant x = 0
    acc.add_point(0.0, 49.9)      # col 0, row 0
    acc.add_point(0.0, 50.0)      # col 0, row 1
    acc.add_point(0.0, 1049.9)    # col 0, row 20
    acc.add_point(0.0, 1050.0)    # col 0, row 21 (partial cell)
    acc.add_point(0.0, 1079.9)    # col 0, row 21 (partial cell)

    data = acc.build()
    assert data.total_points == 10
    assert data.out_of_bounds_points == 0
    assert data.invalid_points == 0

    # Verify X transitions along row 0
    assert data.raw_counts[0, 0] == 2.0   # (49.9, 0.0) and (0.0, 49.9) both land in cell (0, 0)
    assert data.raw_counts[0, 1] == 1.0   # (50.0, 0.0)
    assert data.raw_counts[0, 37] == 1.0  # (1899.9, 0.0)
    assert data.raw_counts[0, 38] == 2.0  # (1900.0, 0.0) and (1919.9, 0.0) in partial cell

    # Verify Y transitions along col 0
    assert data.raw_counts[1, 0] == 1.0   # (0.0, 50.0)
    assert data.raw_counts[20, 0] == 1.0  # (0.0, 1049.9)
    assert data.raw_counts[21, 0] == 2.0  # (0.0, 1050.0) and (0.0, 1079.9) in partial cell


# ====================================================================
# Ingestion Interface Tests
# ====================================================================

def test_add_observations_batch():
    config = HeatmapConfig(image_width=200, image_height=200, cell_size=20)
    acc = HeatmapAccumulator(config)

    obs = [
        FootpointObservation(track_id=1, frame_index=1, x=30.0, y=50.0),
        FootpointObservation(track_id=2, frame_index=1, x=70.0, y=90.0),
        FootpointObservation(track_id=3, frame_index=1, x=-5.0, y=50.0),  # Out of bounds
    ]
    added = acc.add_observations(obs)
    assert added == 2
    assert acc.total_points == 2
    assert acc.out_of_bounds_points == 1


def test_add_trajectory_preserves_gaps():
    """Verify trajectories with gaps contribute only actual observations."""
    config = HeatmapConfig(image_width=200, image_height=200, cell_size=20)
    acc = HeatmapAccumulator(config)

    # Trajectory with gap at frames 12 and 13
    points = (
        FootpointObservation(1, 10, 50.0, 50.0),
        FootpointObservation(1, 11, 52.0, 50.0),
        FootpointObservation(1, 14, 56.0, 50.0),
    )
    traj = Trajectory(track_id=1, points=points)
    assert traj.has_gaps is True

    added = acc.add_trajectory(traj)
    assert added == 3
    assert acc.total_points == 3

    data = acc.build()
    # Exactly 3 points contributed to row 2, col 2
    assert data.raw_counts[2, 2] == 3.0
    assert np.sum(data.raw_counts) == 3.0


def test_accumulator_reset():
    config = HeatmapConfig(image_width=100, image_height=100, cell_size=10)
    acc = HeatmapAccumulator(config)
    acc.add_point(50.0, 50.0)
    acc.add_point(-10.0, 0.0)
    assert acc.total_points == 1
    assert acc.out_of_bounds_points == 1

    acc.reset()
    assert acc.total_points == 0
    assert acc.out_of_bounds_points == 0
    data = acc.build()
    assert np.all(data.raw_counts == 0.0)


# ====================================================================
# Visualization Tests
# ====================================================================

def test_render_heatmap():
    config = HeatmapConfig(image_width=100, image_height=100, cell_size=10)
    acc = HeatmapAccumulator(config)
    acc.add_point(50.0, 50.0)
    data = acc.build()

    img = render_heatmap(data)
    assert img.shape == (100, 100, 3)
    assert img.dtype == np.uint8


def test_render_heatmap_smoothed():
    config = HeatmapConfig(image_width=100, image_height=100, cell_size=10)
    acc = HeatmapAccumulator(config)
    acc.add_point(50.0, 50.0)
    data = acc.build()

    img = render_heatmap(data, sigma=2.0)
    assert img.shape == (100, 100, 3)
    assert img.dtype == np.uint8


def test_overlay_heatmap_does_not_mutate_frame():
    config = HeatmapConfig(image_width=100, image_height=100, cell_size=10)
    acc = HeatmapAccumulator(config)
    acc.add_point(50.0, 50.0)
    data = acc.build()

    heatmap_bgr = render_heatmap(data)
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    frame_copy = frame.copy()

    blended = overlay_heatmap(frame, heatmap_bgr, alpha=0.5, density_map=data.normalized)
    assert blended.shape == frame.shape
    assert blended.dtype == frame.dtype
    # Input frame must remain unmodified
    assert np.array_equal(frame, frame_copy)
    # Output must have blended content
    assert not np.array_equal(blended, frame)
