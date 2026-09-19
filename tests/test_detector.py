"""Tests for person detector interfaces and sequence detection orchestration."""

from pathlib import Path
from typing import Any, List, Union
import pytest
from PIL import Image

from people_analytics.dataset.mot_sequence import MOT17Sequence
from people_analytics.detection.base import BasePersonDetector
from people_analytics.detection.models import DetectionConfig, PersonDetection
from people_analytics.detection.yolo_detector import YOLOv8PersonDetector


class MockPersonDetector(BasePersonDetector):
    """Deterministic mock detector for unit testing without model weights."""

    def __init__(self, config: DetectionConfig, detections_per_frame: List[PersonDetection]):
        super().__init__(config)
        self.detections_per_frame = detections_per_frame

    def detect_frame(
        self,
        image: Union[Path, str, Image.Image, Any],
        frame_index: int,
    ) -> List[PersonDetection]:
        # Filter detections matching this frame and >= confidence_threshold
        return [
            d for d in self.detections_per_frame
            if d.frame_index == frame_index and d.confidence >= self.config.confidence_threshold
        ]


@pytest.fixture
def mock_sequence_with_frames(tmp_path: Path) -> MOT17Sequence:
    seq_dir = tmp_path / "MOT17-00-TEST"
    seq_dir.mkdir()

    ini_content = """[Sequence]
name=MOT17-00-TEST
imDir=img1
frameRate=30
seqLength=3
imWidth=640
imHeight=480
imExt=.jpg
"""
    (seq_dir / "seqinfo.ini").write_text(ini_content, encoding="utf-8")

    img_dir = seq_dir / "img1"
    img_dir.mkdir()
    (img_dir / "000001.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (img_dir / "000002.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (img_dir / "000003.jpg").write_bytes(b"\xff\xd8\xff\xe0")

    return MOT17Sequence(seq_dir)


def test_mock_detector_frame_inference():
    dets = [
        PersonDetection(1, 10.0, 20.0, 30.0, 40.0, 0.9),
        PersonDetection(1, 50.0, 60.0, 30.0, 40.0, 0.3),  # Below 0.5 threshold
        PersonDetection(2, 10.0, 20.0, 30.0, 40.0, 0.8),
    ]
    config = DetectionConfig(confidence_threshold=0.5)
    detector = MockPersonDetector(config, dets)

    # Frame 1: only first detection meets threshold
    f1_dets = detector.detect_frame("dummy_path.jpg", frame_index=1)
    assert len(f1_dets) == 1
    assert f1_dets[0].confidence == 0.9

    # Frame 2: meets threshold
    f2_dets = detector.detect_frame("dummy_path.jpg", frame_index=2)
    assert len(f2_dets) == 1
    assert f2_dets[0].confidence == 0.8

    # Frame 3: no detections
    f3_dets = detector.detect_frame("dummy_path.jpg", frame_index=3)
    assert len(f3_dets) == 0


def test_detect_sequence_orchestration(mock_sequence_with_frames: MOT17Sequence):
    dets = [
        PersonDetection(1, 10.0, 20.0, 30.0, 40.0, 0.9),
        PersonDetection(2, 15.0, 25.0, 30.0, 40.0, 0.8),
        PersonDetection(3, 20.0, 30.0, 30.0, 40.0, 0.7),
    ]
    config = DetectionConfig(confidence_threshold=0.5)
    detector = MockPersonDetector(config, dets)

    # Process all 3 frames
    all_dets = detector.detect_sequence(mock_sequence_with_frames)
    assert len(all_dets) == 3

    # Process subset (frame 1 to 2)
    sub_dets = detector.detect_sequence(mock_sequence_with_frames, start_frame=1, end_frame=2)
    assert len(sub_dets) == 2

    # Process with stride=2 (frame 1, frame 3)
    strided = detector.detect_sequence(mock_sequence_with_frames, stride=2)
    assert len(strided) == 2
    assert {d.frame_index for d in strided} == {1, 3}


def test_detect_sequence_invalid_ranges(mock_sequence_with_frames: MOT17Sequence):
    detector = MockPersonDetector(DetectionConfig(), [])

    with pytest.raises(ValueError, match="start_frame must be >= 1"):
        detector.detect_sequence(mock_sequence_with_frames, start_frame=0)

    with pytest.raises(ValueError, match="start_frame.*must be <= end_frame"):
        detector.detect_sequence(mock_sequence_with_frames, start_frame=3, end_frame=2)


# ====================================================================
# Integration Test with YOLOv8 on Local MOT17 Dataset (isolated)
# ====================================================================

LOCAL_MOT17_09 = Path("MOT17/MOT17/train/MOT17-09-FRCNN")
LOCAL_WEIGHTS = Path("yolov8n.pt")


@pytest.mark.integration
@pytest.mark.skipif(
    not (LOCAL_MOT17_09.exists() and LOCAL_WEIGHTS.exists()),
    reason="Integration test requires local MOT17-09 sequence and pre-downloaded yolov8n.pt weights",
)
def test_yolo_detector_single_frame():
    seq = MOT17Sequence(LOCAL_MOT17_09)
    frame_1_path = seq.get_frame_path(1)

    # Run YOLOv8 on frame 1 using locally available weights
    config = DetectionConfig(confidence_threshold=0.4, model_name=str(LOCAL_WEIGHTS), device="cpu")
    detector = YOLOv8PersonDetector(config)
    detections = detector.detect_frame(frame_1_path, frame_index=1)

    # Frame 1 of MOT17-09 contains pedestrians
    assert len(detections) > 0
    for det in detections:
        assert det.frame_index == 1
        assert det.class_name == "person"
        assert det.confidence >= 0.4
        # Coordinate sanity check against 1920x1080 resolution
        assert 0.0 <= det.bb_left <= 1920.0
        assert 0.0 <= det.bb_top <= 1080.0
        assert det.bb_width > 0.0
        assert det.bb_height > 0.0

