"""Unit and integration tests for footpoint extraction."""

import math
import numpy as np
import pytest

from people_analytics.detection.models import PersonDetection
from people_analytics.footpoint import (
    BaseFootpointExtractor,
    BottomCenterFootpointExtractor,
    Footpoint,
    FootpointObservation,
    draw_footpoints,
)
from people_analytics.tracking.byte_tracker import ByteTracker
from people_analytics.tracking.models import TrackerConfig, TrackObservation, TrackState


# ====================================================================
# Data Model Tests
# ====================================================================

def test_footpoint_properties():
    pt = Footpoint(120.5, 250.0)
    assert pt.x == 120.5
    assert pt.y == 250.0
    assert pt.as_tuple == (120.5, 250.0)


def test_footpoint_immutability():
    pt = Footpoint(10.0, 20.0)
    with pytest.raises(Exception):
        pt.x = 30.0  # dataclass is frozen


def test_footpoint_observation_properties():
    obs = FootpointObservation(
        track_id=42,
        frame_index=15,
        x=100.0,
        y=200.0,
        confidence=0.88,
    )
    assert obs.track_id == 42
    assert obs.frame_index == 15
    assert obs.x == 100.0
    assert obs.y == 200.0
    assert obs.confidence == 0.88
    assert obs.as_tuple == (100.0, 200.0)
    assert obs.point == Footpoint(100.0, 200.0)


# ====================================================================
# Mathematical Extraction Tests
# ====================================================================

def test_bottom_center_deterministic_example():
    """Verify specification deterministic example: (100, 50, 140, 170) -> (120, 170)."""
    extractor = BottomCenterFootpointExtractor()
    bbox = (100.0, 50.0, 140.0, 170.0)
    pt = extractor.extract_point(bbox)
    assert pt.x == 120.0
    assert pt.y == 170.0


def test_bottom_center_float_coordinates():
    extractor = BottomCenterFootpointExtractor()
    bbox = (10.25, 20.5, 30.75, 80.5)
    pt = extractor.extract_point(bbox)
    assert pt.x == pytest.approx(20.5)
    assert pt.y == pytest.approx(80.5)


def test_bottom_center_negative_coordinates():
    extractor = BottomCenterFootpointExtractor()
    bbox = (-40.0, -20.0, -10.0, 50.0)
    pt = extractor.extract_point(bbox)
    assert pt.x == pytest.approx(-25.0)
    assert pt.y == pytest.approx(50.0)


def test_no_silent_clipping_outside_image_boundaries():
    """Verify points exceeding frame boundaries are preserved without artificial clamping."""
    extractor = BottomCenterFootpointExtractor()
    # Box extending below a 1080p frame
    bbox = (1900.0, 1000.0, 1950.0, 1120.0)
    pt = extractor.extract_point(bbox)
    assert pt.x == 1925.0
    assert pt.y == 1120.0  # Must not be clamped to 1080


# ====================================================================
# Validation & Error Handling Tests
# ====================================================================

def test_zero_width_raises_value_error():
    extractor = BottomCenterFootpointExtractor()
    with pytest.raises(ValueError, match="Invalid bounding box width"):
        extractor.extract_point((100.0, 50.0, 100.0, 150.0))


def test_zero_height_raises_value_error():
    extractor = BottomCenterFootpointExtractor()
    with pytest.raises(ValueError, match="Invalid bounding box height"):
        extractor.extract_point((100.0, 50.0, 150.0, 50.0))


def test_negative_dimensions_raise_value_error():
    extractor = BottomCenterFootpointExtractor()
    with pytest.raises(ValueError, match="Invalid bounding box width"):
        extractor.extract_point((150.0, 50.0, 100.0, 150.0))  # x2 < x1

    with pytest.raises(ValueError, match="Invalid bounding box height"):
        extractor.extract_point((100.0, 150.0, 150.0, 50.0))  # y2 < y1


def test_non_finite_coordinates_raise_value_error():
    extractor = BottomCenterFootpointExtractor()
    with pytest.raises(ValueError, match="not finite"):
        extractor.extract_point((float("nan"), 50.0, 150.0, 150.0))

    with pytest.raises(ValueError, match="not finite"):
        extractor.extract_point((100.0, float("inf"), 150.0, 150.0))

    with pytest.raises(ValueError, match="not finite"):
        extractor.extract_point((100.0, 50.0, float("-inf"), 150.0))


# ====================================================================
# TrackObservation Integration Tests
# ====================================================================

def test_extract_from_observation():
    extractor = BottomCenterFootpointExtractor()
    obs = TrackObservation(
        track_id=7,
        frame_index=12,
        bb_left=50.0,
        bb_top=100.0,
        bb_width=60.0,
        bb_height=140.0,
        confidence=0.92,
        state=TrackState.CONFIRMED,
    )

    foot_obs = extractor.extract_from_observation(obs)
    assert foot_obs.track_id == 7
    assert foot_obs.frame_index == 12
    assert foot_obs.x == 80.0   # 50 + 60/2
    assert foot_obs.y == 240.0  # 100 + 140
    assert foot_obs.confidence == 0.92


def test_extract_batch():
    extractor = BottomCenterFootpointExtractor()
    observations = [
        TrackObservation(1, 10, 100.0, 100.0, 40.0, 80.0, 0.9),
        TrackObservation(2, 10, 200.0, 100.0, 50.0, 100.0, 0.85),
    ]
    results = extractor.extract_batch(observations)
    assert len(results) == 2
    assert results[0].track_id == 1
    assert results[0].as_tuple == (120.0, 180.0)
    assert results[1].track_id == 2
    assert results[1].as_tuple == (225.0, 200.0)


def test_extract_batch_empty():
    extractor = BottomCenterFootpointExtractor()
    results = extractor.extract_batch([])
    assert results == []


# ====================================================================
# Lifecycle & LOST Frame Semantic Tests
# ====================================================================

def test_lost_frames_do_not_produce_footpoints():
    """Verify frames where a track is LOST produce zero footpoints.

    Frame 10: Detected -> Footpoint
    Frame 11: Detected -> Footpoint
    Frame 12: Lost     -> No Footpoint
    Frame 13: Lost     -> No Footpoint
    Frame 14: Recovered-> Footpoint
    """
    tracker = ByteTracker(TrackerConfig(min_hits=1, max_lost_frames=5))
    extractor = BottomCenterFootpointExtractor()

    # Frame 10: Detected
    d10 = [PersonDetection(10, 100.0, 100.0, 50.0, 100.0, 0.9)]
    obs10 = tracker.update(d10, frame_index=10)
    fp10 = extractor.extract_batch(obs10)
    assert len(fp10) == 1
    assert fp10[0].track_id == 1
    assert fp10[0].frame_index == 10
    assert fp10[0].as_tuple == (125.0, 200.0)

    # Frame 11: Detected
    d11 = [PersonDetection(11, 102.0, 100.0, 50.0, 100.0, 0.9)]
    obs11 = tracker.update(d11, frame_index=11)
    fp11 = extractor.extract_batch(obs11)
    assert len(fp11) == 1
    assert fp11[0].track_id == 1
    assert fp11[0].frame_index == 11
    assert fp11[0].as_tuple == (127.0, 200.0)

    # Frame 12: Temporarily Lost -> zero observations -> ZERO footpoints
    obs12 = tracker.update([], frame_index=12)
    fp12 = extractor.extract_batch(obs12)
    assert len(obs12) == 0
    assert len(fp12) == 0

    # Frame 13: Temporarily Lost -> zero observations -> ZERO footpoints
    obs13 = tracker.update([], frame_index=13)
    fp13 = extractor.extract_batch(obs13)
    assert len(obs13) == 0
    assert len(fp13) == 0

    # Frame 14: Recovered
    d14 = [PersonDetection(14, 106.0, 100.0, 50.0, 100.0, 0.85)]
    obs14 = tracker.update(d14, frame_index=14)
    fp14 = extractor.extract_batch(obs14)
    assert len(fp14) == 1
    assert fp14[0].track_id == 1
    assert fp14[0].frame_index == 14
    assert fp14[0].as_tuple == (131.0, 200.0)


# ====================================================================
# Minimal Visualization Tests
# ====================================================================

def test_draw_footpoints():
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    observations = [
        FootpointObservation(track_id=1, frame_index=1, x=100.0, y=200.0),
        FootpointObservation(track_id=2, frame_index=1, x=300.0, y=400.0),
    ]
    annotated = draw_footpoints(canvas, observations)
    assert annotated.shape == canvas.shape
    assert annotated.dtype == canvas.dtype
    # Check that pixels were modified
    assert not np.array_equal(canvas, annotated)


def test_draw_footpoints_empty():
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = draw_footpoints(canvas, [])
    assert np.array_equal(canvas, annotated)
