"""Person detection subsystem."""

from people_analytics.detection.base import BasePersonDetector
from people_analytics.detection.models import DetectionConfig, PersonDetection
from people_analytics.detection.yolo_detector import YOLOv8PersonDetector

__all__ = [
    "BasePersonDetector",
    "DetectionConfig",
    "PersonDetection",
    "YOLOv8PersonDetector",
]
