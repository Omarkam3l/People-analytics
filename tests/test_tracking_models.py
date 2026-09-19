"""Unit tests for tracking data models and configurations."""

import pytest

from people_analytics.tracking.models import (
    TrackerConfig,
    TrackObservation,
    TrackState,
)


def test_track_observation_properties():
    obs = TrackObservation(
        track_id=42,
        frame_index=10,
        bb_left=100.0,
        bb_top=150.0,
        bb_width=50.0,
        bb_height=120.0,
        confidence=0.88,
        state=TrackState.CONFIRMED,
    )

    assert obs.track_id == 42
    assert obs.frame_index == 10
    assert obs.bbox_xywh == (100.0, 150.0, 50.0, 120.0)
    assert obs.bbox_xyxy == (100.0, 150.0, 150.0, 270.0)
    assert obs.confidence == 0.88
    assert obs.state == TrackState.CONFIRMED


def test_track_observation_immutability():
    obs = TrackObservation(
        track_id=1,
        frame_index=1,
        bb_left=0.0,
        bb_top=0.0,
        bb_width=10.0,
        bb_height=20.0,
        confidence=0.9,
    )
    with pytest.raises(AttributeError):
        obs.track_id = 2  # type: ignore


def test_tracker_config_defaults():
    config = TrackerConfig()
    assert config.high_threshold == 0.5
    assert config.low_threshold == 0.1
    assert config.match_threshold == 0.5
    assert config.min_hits == 2
    assert config.max_lost_frames == 30
