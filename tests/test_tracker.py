"""Unit and integration tests for multi-object tracking."""

from pathlib import Path
import pytest

from people_analytics.dataset.mot_sequence import MOT17Sequence
from people_analytics.detection.models import DetectionConfig, PersonDetection
from people_analytics.detection.yolo_detector import YOLOv8PersonDetector
from people_analytics.tracking.byte_tracker import ByteTracker
from people_analytics.tracking.models import TrackerConfig, TrackState


def test_first_track_creation_and_confirmation():
    # min_hits = 2: first frame creates tentative, second confirms
    tracker = ByteTracker(TrackerConfig(min_hits=2, high_threshold=0.5))
    det1 = [PersonDetection(frame_index=1, bb_left=100.0, bb_top=100.0, bb_width=50.0, bb_height=100.0, confidence=0.9)]

    # Frame 1: Tentative track created, not yet emitted
    obs1 = tracker.update(det1, frame_index=1)
    assert len(obs1) == 0
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].state == TrackState.TENTATIVE
    assert tracker.tracks[0].track_id == 1

    # Frame 2: Second hit confirms track, observation emitted
    det2 = [PersonDetection(frame_index=2, bb_left=102.0, bb_top=100.0, bb_width=50.0, bb_height=100.0, confidence=0.85)]
    obs2 = tracker.update(det2, frame_index=2)
    assert len(obs2) == 1
    assert obs2[0].track_id == 1
    assert obs2[0].state == TrackState.CONFIRMED


def test_identity_persistence():
    # Track a moving pedestrian across 5 frames
    tracker = ByteTracker(TrackerConfig(min_hits=1, high_threshold=0.5))

    for frame in range(1, 6):
        x = 100.0 + (frame - 1) * 5.0
        dets = [PersonDetection(frame_index=frame, bb_left=x, bb_top=50.0, bb_width=40.0, bb_height=80.0, confidence=0.9)]
        obs = tracker.update(dets, frame_index=frame)
        assert len(obs) == 1
        assert obs[0].track_id == 1
        assert obs[0].bb_left == x


def test_multiple_simultaneous_tracks():
    # 2 pedestrians moving simultaneously
    tracker = ByteTracker(TrackerConfig(min_hits=1, high_threshold=0.5))

    for frame in range(1, 4):
        dets = [
            PersonDetection(frame_index=frame, bb_left=50.0 + frame, bb_top=50.0, bb_width=30.0, bb_height=70.0, confidence=0.9),
            PersonDetection(frame_index=frame, bb_left=300.0 + frame, bb_top=50.0, bb_width=30.0, bb_height=70.0, confidence=0.85),
        ]
        obs = tracker.update(dets, frame_index=frame)
        assert len(obs) == 2
        ids = {o.track_id for o in obs}
        assert ids == {1, 2}


def test_track_occlusion_and_low_score_recovery():
    # ByteTrack stage 2: target occluded (confidence drops to 0.25), recovered in stage 2
    tracker = ByteTracker(TrackerConfig(min_hits=1, high_threshold=0.5, low_threshold=0.1, match_threshold=0.5))

    # Frame 1: High confidence detection -> confirmed track 1
    d1 = [PersonDetection(1, 100.0, 100.0, 50.0, 100.0, 0.9)]
    obs1 = tracker.update(d1, frame_index=1)
    assert len(obs1) == 1
    assert obs1[0].track_id == 1

    # Frame 2: Missing detection -> track becomes LOST
    obs2 = tracker.update([], frame_index=2)
    assert len(obs2) == 0
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].state == TrackState.LOST

    # Frame 3: Low-confidence detection (0.25) -> stage 2 recovers track 1!
    d3 = [PersonDetection(3, 102.0, 100.0, 50.0, 100.0, 0.25)]
    obs3 = tracker.update(d3, frame_index=3)
    assert len(obs3) == 1
    assert obs3[0].track_id == 1
    assert obs3[0].state == TrackState.CONFIRMED


def test_track_termination_after_max_lost_frames():
    tracker = ByteTracker(TrackerConfig(min_hits=1, high_threshold=0.5, max_lost_frames=2))

    # Frame 1: Detection -> track 1
    tracker.update([PersonDetection(1, 50.0, 50.0, 30.0, 60.0, 0.9)], frame_index=1)
    assert len(tracker.tracks) == 1

    # Frame 2 & 3: Missing (lost_frames = 1, then 2)
    tracker.update([], frame_index=2)
    assert tracker.tracks[0].state == TrackState.LOST
    tracker.update([], frame_index=3)
    assert tracker.tracks[0].state == TrackState.LOST

    # Frame 4: Missing (lost_frames = 3 > max_lost_frames 2) -> TERMINATED & purged
    tracker.update([], frame_index=4)
    assert len(tracker.tracks) == 0


def test_monotonic_id_no_reuse():
    tracker = ByteTracker(TrackerConfig(min_hits=1, max_lost_frames=1))

    # Target 1 created
    tracker.update([PersonDetection(1, 10.0, 10.0, 20.0, 40.0, 0.9)], frame_index=1)
    assert tracker.tracks[0].track_id == 1

    # Target 1 lost and terminated
    tracker.update([], frame_index=2)
    tracker.update([], frame_index=3)
    assert len(tracker.tracks) == 0

    # New target appears -> MUST receive ID 2, never re-using ID 1
    tracker.update([PersonDetection(4, 500.0, 500.0, 20.0, 40.0, 0.9)], frame_index=4)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == 2


def test_empty_frames_handling():
    tracker = ByteTracker(TrackerConfig())
    obs = tracker.update([], frame_index=1)
    assert obs == []
    assert len(tracker.tracks) == 0


def test_tracker_reset():
    tracker = ByteTracker(TrackerConfig(min_hits=1))
    tracker.update([PersonDetection(1, 10.0, 10.0, 20.0, 40.0, 0.9)], frame_index=1)
    assert len(tracker.tracks) == 1

    tracker.reset()
    assert len(tracker.tracks) == 0

    # After reset, new track starts at ID 1 again
    tracker.update([PersonDetection(1, 50.0, 50.0, 20.0, 40.0, 0.9)], frame_index=1)
    assert tracker.tracks[0].track_id == 1


def test_lost_state_does_not_create_fake_spatial_observations():
    """Verify that when a track is temporarily lost (frames 12 & 13),
    it remains in LOST state internally but emits ZERO TrackObservation objects,
    preventing fabricated spatial observations."""
    tracker = ByteTracker(TrackerConfig(min_hits=1, max_lost_frames=5))

    # Frame 10: Detected -> emitted
    d10 = [PersonDetection(10, 100.0, 100.0, 50.0, 100.0, 0.9)]
    obs10 = tracker.update(d10, frame_index=10)
    assert len(obs10) == 1
    assert obs10[0].track_id == 1
    assert obs10[0].frame_index == 10

    # Frame 11: Detected -> emitted
    d11 = [PersonDetection(11, 102.0, 100.0, 50.0, 100.0, 0.9)]
    obs11 = tracker.update(d11, frame_index=11)
    assert len(obs11) == 1
    assert obs11[0].track_id == 1
    assert obs11[0].frame_index == 11

    # Frame 12: Temporarily lost -> ZERO observations emitted
    obs12 = tracker.update([], frame_index=12)
    assert len(obs12) == 0
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].state == TrackState.LOST
    assert tracker.tracks[0].track_id == 1

    # Frame 13: Temporarily lost -> ZERO observations emitted
    obs13 = tracker.update([], frame_index=13)
    assert len(obs13) == 0
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].state == TrackState.LOST
    assert tracker.tracks[0].track_id == 1

    # Frame 14: Detected again -> recovered and emitted with same track ID 1
    d14 = [PersonDetection(14, 106.0, 100.0, 50.0, 100.0, 0.85)]
    obs14 = tracker.update(d14, frame_index=14)
    assert len(obs14) == 1
    assert obs14[0].track_id == 1
    assert obs14[0].frame_index == 14
    assert obs14[0].state == TrackState.CONFIRMED


# ====================================================================
# Integration Test: Detector -> Tracker Pipeline
# ====================================================================

LOCAL_MOT17_09 = Path("MOT17/MOT17/train/MOT17-09-FRCNN")
LOCAL_WEIGHTS = Path("yolov8n.pt")


@pytest.mark.integration
@pytest.mark.skipif(
    not (LOCAL_MOT17_09.exists() and LOCAL_WEIGHTS.exists()),
    reason="Integration test requires local sequence and yolov8n.pt weights",
)
def test_detector_to_tracker_pipeline():
    seq = MOT17Sequence(LOCAL_MOT17_09)
    det_config = DetectionConfig(confidence_threshold=0.4, model_name=str(LOCAL_WEIGHTS), device="cpu")
    detector = YOLOv8PersonDetector(det_config)

    trk_config = TrackerConfig(high_threshold=0.4, min_hits=2)
    tracker = ByteTracker(trk_config)

    all_tracks = []
    for frame_idx in range(1, 4):
        img_path = seq.get_frame_path(frame_idx)
        dets = detector.detect_frame(img_path, frame_index=frame_idx)
        tracks = tracker.update(dets, frame_index=frame_idx)
        all_tracks.extend(tracks)

    # By frame 3, multiple confirmed tracks should be active
    assert len(all_tracks) > 0
    unique_ids = set(t.track_id for t in all_tracks)
    assert len(unique_ids) > 0
    for trk in all_tracks:
        assert trk.state == TrackState.CONFIRMED
        assert 0.0 <= trk.bb_left <= 1920.0
        assert 0.0 <= trk.bb_top <= 1080.0
