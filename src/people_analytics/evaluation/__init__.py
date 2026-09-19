"""Evaluation subsystem."""

from people_analytics.evaluation.detection_metrics import (
    DetectionMetrics,
    compute_iou,
    evaluate_detections_frame,
    evaluate_sequence_detections,
)
from people_analytics.evaluation.tracking_metrics import (
    TrackingMetrics,
    evaluate_tracking_sequence,
)

__all__ = [
    "DetectionMetrics",
    "TrackingMetrics",
    "compute_iou",
    "evaluate_detections_frame",
    "evaluate_sequence_detections",
    "evaluate_tracking_sequence",
]

