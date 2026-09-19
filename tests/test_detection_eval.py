"""Tests for detection evaluation metrics and IoU calculation."""

import pytest

from people_analytics.dataset.models import GroundTruthAnnotation
from people_analytics.detection.models import PersonDetection
from people_analytics.evaluation.detection_metrics import (
    compute_iou,
    evaluate_detections_frame,
    evaluate_sequence_detections,
)


def test_compute_iou_identical():
    box = (100.0, 100.0, 200.0, 200.0)
    assert compute_iou(box, box) == 1.0


def test_compute_iou_disjoint():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (20.0, 20.0, 30.0, 30.0)
    assert compute_iou(box1, box2) == 0.0


def test_compute_iou_partial_overlap():
    # box1: (0, 0, 10, 10) -> area 100
    # box2: (5, 0, 15, 10) -> area 100
    # intersection: (5, 0, 10, 10) -> area 50
    # union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 1/3
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (5.0, 0.0, 15.0, 10.0)
    assert pytest.approx(compute_iou(box1, box2), 1e-5) == 1.0 / 3.0


def test_compute_iou_zero_area():
    box1 = (0.0, 0.0, 0.0, 10.0)
    box2 = (0.0, 0.0, 10.0, 10.0)
    assert compute_iou(box1, box2) == 0.0


def test_evaluate_detections_frame_all_tp():
    # 2 detections perfectly matching 2 GT pedestrians
    dets = [
        PersonDetection(1, 10.0, 10.0, 20.0, 40.0, 0.9),
        PersonDetection(1, 100.0, 100.0, 30.0, 60.0, 0.8),
    ]
    gt = [
        GroundTruthAnnotation(1, 1, 10.0, 10.0, 20.0, 40.0, 1.0, 1, 1.0),
        GroundTruthAnnotation(1, 2, 100.0, 100.0, 30.0, 60.0, 1.0, 1, 1.0),
    ]
    metrics = evaluate_detections_frame(dets, gt, iou_threshold=0.5)
    assert metrics.true_positives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0


def test_evaluate_detections_frame_with_fp_and_fn():
    # 1 TP match, 1 FP (spurious det), 1 FN (missed GT)
    dets = [
        PersonDetection(1, 10.0, 10.0, 20.0, 40.0, 0.9),  # matches gt[0]
        PersonDetection(1, 500.0, 500.0, 20.0, 40.0, 0.7),  # spurious
    ]
    gt = [
        GroundTruthAnnotation(1, 1, 10.0, 10.0, 20.0, 40.0, 1.0, 1, 1.0),  # matched
        GroundTruthAnnotation(1, 2, 200.0, 200.0, 20.0, 40.0, 1.0, 1, 1.0),  # missed
    ]
    metrics = evaluate_detections_frame(dets, gt, iou_threshold=0.5)
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1_score == 0.5


def test_evaluate_detections_frame_ignores_inactive_and_non_pedestrian():
    dets = [
        PersonDetection(1, 10.0, 10.0, 20.0, 40.0, 0.9),
    ]
    gt = [
        # Inactive pedestrian (conf == 0.0) -> should be ignored
        GroundTruthAnnotation(1, 1, 10.0, 10.0, 20.0, 40.0, 0.0, 1, 1.0),
        # Distractor (class_id == 8) -> should be ignored
        GroundTruthAnnotation(1, 2, 10.0, 10.0, 20.0, 40.0, 1.0, 8, 1.0),
    ]
    metrics = evaluate_detections_frame(dets, gt, iou_threshold=0.5)
    # Since active GT is empty, detection is treated as False Positive
    assert metrics.true_positives == 0
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.precision == 0.0


def test_evaluate_detections_frame_empty():
    # Both empty
    m_empty = evaluate_detections_frame([], [])
    assert m_empty.precision == 1.0
    assert m_empty.recall == 1.0
    assert m_empty.f1_score == 1.0

    # No detections, 1 GT
    gt = [GroundTruthAnnotation(1, 1, 10.0, 10.0, 20.0, 40.0, 1.0, 1, 1.0)]
    m_miss = evaluate_detections_frame([], gt)
    assert m_miss.true_positives == 0
    assert m_miss.false_negatives == 1
    assert m_miss.recall == 0.0


def test_evaluate_sequence_detections():
    # Frame 1: 1 TP
    # Frame 2: 1 TP, 1 FP
    dets = [
        PersonDetection(1, 10.0, 10.0, 20.0, 40.0, 0.9),
        PersonDetection(2, 15.0, 15.0, 20.0, 40.0, 0.85),
        PersonDetection(2, 500.0, 500.0, 20.0, 40.0, 0.7),
    ]
    gt = [
        GroundTruthAnnotation(1, 1, 10.0, 10.0, 20.0, 40.0, 1.0, 1, 1.0),
        GroundTruthAnnotation(2, 1, 15.0, 15.0, 20.0, 40.0, 1.0, 1, 1.0),
    ]
    metrics = evaluate_sequence_detections(dets, gt, iou_threshold=0.5)
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert pytest.approx(metrics.precision, 1e-3) == 2.0 / 3.0
    assert metrics.recall == 1.0
    assert pytest.approx(metrics.f1_score, 1e-3) == 0.8
