"""Tests for detection data models and configurations."""

import pytest

from people_analytics.detection.models import DetectionConfig, PersonDetection


def test_person_detection_properties():
    det = PersonDetection(
        frame_index=5,
        bb_left=100.0,
        bb_top=200.0,
        bb_width=50.0,
        bb_height=120.0,
        confidence=0.85,
        class_name="person",
    )

    assert det.frame_index == 5
    assert det.bbox_xywh == (100.0, 200.0, 50.0, 120.0)
    assert det.bbox_xyxy == (100.0, 200.0, 150.0, 320.0)
    assert det.area == 6000.0
    assert det.confidence == 0.85
    assert det.class_name == "person"


def test_person_detection_immutability():
    det = PersonDetection(
        frame_index=1,
        bb_left=0.0,
        bb_top=0.0,
        bb_width=10.0,
        bb_height=20.0,
        confidence=0.9,
    )
    with pytest.raises(AttributeError):
        det.confidence = 0.5  # type: ignore


def test_detection_config_defaults():
    config = DetectionConfig()
    assert config.confidence_threshold == 0.5
    assert config.iou_threshold == 0.5
    assert config.model_name == "yolov8n.pt"
    assert config.device == "cpu"
