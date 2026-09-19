"""Unit and integration tests for trajectory construction."""

import numpy as np
import pytest

from people_analytics.detection.models import PersonDetection
from people_analytics.footpoint.extractor import BottomCenterFootpointExtractor
from people_analytics.footpoint.models import FootpointObservation
from people_analytics.tracking.byte_tracker import ByteTracker
from people_analytics.tracking.models import TrackerConfig, TrackObservation
from people_analytics.trajectory import Trajectory, TrajectoryBuilder, draw_trajectories


# ====================================================================
# Data Model Tests
# ====================================================================

def test_empty_trajectory():
    traj = Trajectory(track_id=1)
    assert len(traj) == 0
    assert traj.is_empty is True
    assert traj.start_frame is None
    assert traj.end_frame is None
    assert traj.frame_indices == ()
    assert traj.coordinates == ()
    assert traj.has_gaps is False
    assert traj.get_point_at_frame(1) is None


def test_single_point_trajectory():
    p = FootpointObservation(track_id=5, frame_index=10, x=100.0, y=200.0)
    traj = Trajectory(track_id=5, points=(p,))
    assert len(traj) == 1
    assert traj.is_empty is False
    assert traj.start_frame == 10
    assert traj.end_frame == 10
    assert traj.frame_indices == (10,)
    assert traj.coordinates == ((100.0, 200.0),)
    assert traj.has_gaps is False
    assert traj.get_point_at_frame(10) == p
    assert traj.get_point_at_frame(11) is None


def test_trajectory_multi_point_sequence():
    points = (
        FootpointObservation(track_id=1, frame_index=1, x=10.0, y=20.0),
        FootpointObservation(track_id=1, frame_index=2, x=12.0, y=22.0),
        FootpointObservation(track_id=1, frame_index=3, x=15.0, y=25.0),
    )
    traj = Trajectory(track_id=1, points=points)
    assert len(traj) == 3
    assert traj.start_frame == 1
    assert traj.end_frame == 3
    assert traj.frame_indices == (1, 2, 3)
    assert traj.coordinates == ((10.0, 20.0), (12.0, 22.0), (15.0, 25.0))
    assert traj.has_gaps is False
    assert list(traj) == list(points)
    assert traj[1] == points[1]


# ====================================================================
# Invariants & Validation Tests
# ====================================================================

def test_mismatched_track_id_raises_value_error():
    points = (
        FootpointObservation(track_id=1, frame_index=1, x=10.0, y=20.0),
        FootpointObservation(track_id=2, frame_index=2, x=12.0, y=22.0),  # Mismatch
    )
    with pytest.raises(ValueError, match="has track_id 2, expected 1"):
        Trajectory(track_id=1, points=points)


def test_duplicate_frame_index_raises_value_error():
    points = (
        FootpointObservation(track_id=1, frame_index=5, x=10.0, y=20.0),
        FootpointObservation(track_id=1, frame_index=5, x=15.0, y=25.0),  # Duplicate
    )
    with pytest.raises(ValueError, match="Duplicate observation for track_id 1 at frame 5"):
        Trajectory(track_id=1, points=points)


def test_non_chronological_points_raise_value_error():
    points = (
        FootpointObservation(track_id=1, frame_index=10, x=10.0, y=20.0),
        FootpointObservation(track_id=1, frame_index=5, x=12.0, y=22.0),  # Out of order
    )
    with pytest.raises(ValueError, match="must be strictly chronological"):
        Trajectory(track_id=1, points=points)


# ====================================================================
# Gap Semantics Tests
# ====================================================================

def test_gap_preservation_semantics():
    """Verify frames 12 and 13 missing from [10, 11, 14] are preserved as gaps."""
    p10 = FootpointObservation(track_id=3, frame_index=10, x=50.0, y=100.0)
    p11 = FootpointObservation(track_id=3, frame_index=11, x=52.0, y=100.0)
    p14 = FootpointObservation(track_id=3, frame_index=14, x=58.0, y=102.0)
    traj = Trajectory(track_id=3, points=(p10, p11, p14))

    assert len(traj) == 3
    assert traj.has_gaps is True
    assert traj.start_frame == 10
    assert traj.end_frame == 14

    # Direct point queries
    assert traj.get_point_at_frame(10) == p10
    assert traj.get_point_at_frame(11) == p11
    assert traj.get_point_at_frame(12) is None  # Preserved as gap
    assert traj.get_point_at_frame(13) is None  # Preserved as gap
    assert traj.get_point_at_frame(14) == p14
    assert traj.get_point_at_frame(15) is None


def test_has_gaps_semantics_definition():
    """Verify exact has_gaps boolean semantics:
    - [10, 11, 14] -> True
    - [10, 11, 12] -> False
    """
    p10 = FootpointObservation(1, 10, 50.0, 100.0)
    p11 = FootpointObservation(1, 11, 52.0, 100.0)
    p12 = FootpointObservation(1, 12, 54.0, 100.0)
    p14 = FootpointObservation(1, 14, 58.0, 102.0)

    # Gap present
    traj_with_gap = Trajectory(1, (p10, p11, p14))
    assert traj_with_gap.has_gaps is True

    # No gap (contiguous)
    traj_contiguous = Trajectory(1, (p10, p11, p12))
    assert traj_contiguous.has_gaps is False


def test_get_point_at_frame_exact_lookup_no_interpolation():
    """Verify get_point_at_frame is strictly an exact match lookup:
    - Existing frame returns FootpointObservation
    - Missing frame returns None (must NOT return nearest or interpolate)
    """
    p1 = FootpointObservation(1, 10, 100.0, 200.0)
    p2 = FootpointObservation(1, 20, 200.0, 300.0)
    traj = Trajectory(1, (p1, p2))

    # Exact matches
    assert traj.get_point_at_frame(10) == p1
    assert traj.get_point_at_frame(20) == p2

    # In-between missing frames: must return None, NOT interpolated (150, 250) or nearest p1/p2
    assert traj.get_point_at_frame(11) is None
    assert traj.get_point_at_frame(15) is None
    assert traj.get_point_at_frame(19) is None

    # Before start and after end
    assert traj.get_point_at_frame(9) is None
    assert traj.get_point_at_frame(21) is None


# ====================================================================
# Timestamp Derivation Tests
# ====================================================================

def test_timestamp_derivation():
    # 30 FPS: Frame 1 is 0.0s, Frame 31 is 1.0s, Frame 46 is 1.5s
    assert Trajectory.get_timestamp(frame_index=1, fps=30.0) == pytest.approx(0.0)
    assert Trajectory.get_timestamp(frame_index=31, fps=30.0) == pytest.approx(1.0)
    assert Trajectory.get_timestamp(frame_index=46, fps=30.0) == pytest.approx(1.5)


def test_timestamp_invalid_fps():
    with pytest.raises(ValueError, match="FPS must be strictly positive"):
        Trajectory.get_timestamp(frame_index=10, fps=0.0)

    with pytest.raises(ValueError, match="FPS must be strictly positive"):
        Trajectory.get_timestamp(frame_index=10, fps=-25.0)


# ====================================================================
# TrajectoryBuilder Tests
# ====================================================================

def test_builder_single_track():
    builder = TrajectoryBuilder()
    builder.add_observation(FootpointObservation(1, 1, 10.0, 20.0))
    builder.add_observation(FootpointObservation(1, 2, 12.0, 22.0))

    traj = builder.build_trajectory(1)
    assert traj.track_id == 1
    assert len(traj) == 2
    assert traj.frame_indices == (1, 2)


def test_out_of_order_builder_input_produces_chronological_trajectory():
    """Verify builder accepts arbitrary input order and produces strictly chronological Trajectory."""
    builder = TrajectoryBuilder()
    # Add frames in scrambled order: 15, 3, 42, 1, 8
    scrambled = [
        FootpointObservation(1, 15, 150.0, 250.0),
        FootpointObservation(1, 3, 30.0, 50.0),
        FootpointObservation(1, 42, 420.0, 600.0),
        FootpointObservation(1, 1, 10.0, 20.0),
        FootpointObservation(1, 8, 80.0, 120.0),
    ]
    for obs in scrambled:
        builder.add_observation(obs)

    traj = builder.build_trajectory(1)
    # Output must be strictly chronological
    assert traj.frame_indices == (1, 3, 8, 15, 42)
    assert traj.coordinates == (
        (10.0, 20.0),
        (30.0, 50.0),
        (80.0, 120.0),
        (150.0, 250.0),
        (420.0, 600.0),
    )


def test_builder_multi_track_batch():
    builder = TrajectoryBuilder()
    batch = [
        FootpointObservation(1, 1, 10.0, 20.0),
        FootpointObservation(2, 1, 100.0, 200.0),
        FootpointObservation(1, 2, 12.0, 22.0),
        FootpointObservation(2, 2, 105.0, 205.0),
        FootpointObservation(3, 2, 500.0, 600.0),
    ]
    builder.add_observations(batch)
    assert builder.track_ids == [1, 2, 3]

    all_trajs = builder.build_all()
    assert len(all_trajs) == 3
    assert len(all_trajs[1]) == 2
    assert len(all_trajs[2]) == 2
    assert len(all_trajs[3]) == 1


def test_builder_duplicate_frame_raises_value_error():
    builder = TrajectoryBuilder()
    builder.add_observation(FootpointObservation(1, 5, 10.0, 20.0))
    builder.add_observation(FootpointObservation(1, 5, 12.0, 22.0))  # Duplicate

    with pytest.raises(ValueError, match="Duplicate observation"):
        builder.build_trajectory(1)


def test_builder_unknown_track_raises_key_error():
    builder = TrajectoryBuilder()
    with pytest.raises(KeyError, match="has no recorded observations"):
        builder.build_trajectory(99)


def test_builder_clear():
    builder = TrajectoryBuilder()
    builder.add_observation(FootpointObservation(1, 1, 10.0, 20.0))
    assert len(builder.track_ids) == 1
    builder.clear()
    assert len(builder.track_ids) == 0


# ====================================================================
# End-to-End Pipeline Integration Test
# ====================================================================

def test_pipeline_tracker_to_footpoint_to_trajectory():
    """Verify TrackObservation -> FootpointObservation -> Trajectory pipeline.

    Evaluates frames 10, 11, 12 (lost), 13 (lost), 14 (recovered) sequence.
    """
    tracker = ByteTracker(TrackerConfig(min_hits=1, max_lost_frames=5))
    extractor = BottomCenterFootpointExtractor()
    builder = TrajectoryBuilder()

    # Frame 10: Detected
    obs10 = tracker.update([PersonDetection(10, 100.0, 100.0, 40.0, 80.0, 0.9)], frame_index=10)
    fp10 = extractor.extract_batch(obs10)
    builder.add_observations(fp10)

    # Frame 11: Detected
    obs11 = tracker.update([PersonDetection(11, 102.0, 100.0, 40.0, 80.0, 0.9)], frame_index=11)
    fp11 = extractor.extract_batch(obs11)
    builder.add_observations(fp11)

    # Frame 12: Temporarily Lost -> 0 observations
    obs12 = tracker.update([], frame_index=12)
    fp12 = extractor.extract_batch(obs12)
    builder.add_observations(fp12)

    # Frame 13: Temporarily Lost -> 0 observations
    obs13 = tracker.update([], frame_index=13)
    fp13 = extractor.extract_batch(obs13)
    builder.add_observations(fp13)

    # Frame 14: Recovered
    obs14 = tracker.update([PersonDetection(14, 108.0, 100.0, 40.0, 80.0, 0.88)], frame_index=14)
    fp14 = extractor.extract_batch(obs14)
    builder.add_observations(fp14)

    # Build final trajectory
    traj = builder.build_trajectory(track_id=1)

    assert traj.track_id == 1
    assert len(traj) == 3
    assert traj.frame_indices == (10, 11, 14)
    assert traj.has_gaps is True
    assert traj.get_point_at_frame(10).as_tuple == (120.0, 180.0)
    assert traj.get_point_at_frame(11).as_tuple == (122.0, 180.0)
    assert traj.get_point_at_frame(12) is None
    assert traj.get_point_at_frame(13) is None
    assert traj.get_point_at_frame(14).as_tuple == (128.0, 180.0)


# ====================================================================
# Minimal Visualization Tests
# ====================================================================

def test_draw_trajectories():
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    points = (
        FootpointObservation(1, 1, 50.0, 50.0),
        FootpointObservation(1, 2, 60.0, 60.0),
        FootpointObservation(1, 3, 70.0, 70.0),
    )
    traj = Trajectory(1, points)
    annotated = draw_trajectories(canvas, [traj], current_frame=3)
    assert annotated.shape == canvas.shape
    assert not np.array_equal(canvas, annotated)


def test_draw_trajectories_respects_gap():
    """Verify gap lines are not drawn across missing frames."""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    # Gap between frame 1 and frame 10
    points = (
        FootpointObservation(1, 1, 50.0, 50.0),
        FootpointObservation(1, 10, 60.0, 60.0),
    )
    traj = Trajectory(1, points)
    annotated = draw_trajectories(canvas, [traj], gap_threshold=1)
    # Because gap > 1, no line should be drawn
    assert np.array_equal(canvas, annotated)
