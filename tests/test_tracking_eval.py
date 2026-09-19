"""Unit tests for tracking evaluation metrics (MOTA, IDF1, ID switches)."""

import pytest

from people_analytics.dataset.models import GroundTruthAnnotation
from people_analytics.evaluation.tracking_metrics import (
    TrackingMetrics,
    evaluate_tracking_sequence,
)
from people_analytics.tracking.models import TrackObservation, TrackState


def test_evaluate_tracking_sequence_perfect():
    # 2 pedestrians over 2 frames, perfectly tracked with stable IDs
    tracks = [
        TrackObservation(1, 1, 10.0, 10.0, 20.0, 40.0, 0.9),
        TrackObservation(2, 1, 100.0, 100.0, 30.0, 60.0, 0.85),
        TrackObservation(1, 2, 12.0, 10.0, 20.0, 40.0, 0.92),
        TrackObservation(2, 2, 102.0, 100.0, 30.0, 60.0, 0.88),
    ]
    gt = [
        GroundTruthAnnotation(1, 101, 10.0, 10.0, 20.0, 40.0, 1.0, 1, 1.0),
        GroundTruthAnnotation(1, 102, 100.0, 100.0, 30.0, 60.0, 1.0, 1, 1.0),
        GroundTruthAnnotation(2, 101, 12.0, 10.0, 20.0, 40.0, 1.0, 1, 1.0),
        GroundTruthAnnotation(2, 102, 102.0, 100.0, 30.0, 60.0, 1.0, 1, 1.0),
    ]

    metrics = evaluate_tracking_sequence(tracks, gt, iou_threshold=0.5)
    assert metrics.total_gt == 4
    assert metrics.true_positives == 4
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.id_switches == 0
    assert metrics.mota == 1.0
    assert metrics.idf1 == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_evaluate_tracking_sequence_with_id_switch():
    # Target 101 tracked by track_id=1 in frame 1, but track_id=2 in frame 2
    tracks = [
        TrackObservation(1, 1, 10.0, 10.0, 20.0, 40.0, 0.9),
        TrackObservation(2, 2, 12.0, 10.0, 20.0, 40.0, 0.9),  # Switched ID!
    ]
    gt = [
        GroundTruthAnnotation(1, 101, 10.0, 10.0, 20.0, 40.0, 1.0, 1, 1.0),
        GroundTruthAnnotation(2, 101, 12.0, 10.0, 20.0, 40.0, 1.0, 1, 1.0),
    ]

    metrics = evaluate_tracking_sequence(tracks, gt, iou_threshold=0.5)
    assert metrics.total_gt == 2
    assert metrics.true_positives == 2
    assert metrics.id_switches == 1
    # MOTA = 1 - (0 + 0 + 1)/2 = 0.5
    assert metrics.mota == 0.5
    # IDF1 < 1.0 due to identity switch
    assert metrics.idf1 < 1.0


def test_evaluate_tracking_sequence_empty():
    empty_metrics = evaluate_tracking_sequence([], [])
    assert empty_metrics.total_gt == 0
    assert empty_metrics.mota == 1.0
    assert empty_metrics.idf1 == 1.0


def test_evaluate_tracking_sequence_all_fp():
    # Tracks exist, but zero GT
    tracks = [TrackObservation(1, 1, 10.0, 10.0, 20.0, 40.0, 0.9)]
    metrics = evaluate_tracking_sequence(tracks, [])
    assert metrics.total_gt == 0
    assert metrics.false_positives == 1
    assert metrics.true_positives == 0
    assert metrics.mota == 0.0
    assert metrics.idf1 == 0.0
    assert metrics.precision == 0.0


def test_evaluate_tracking_sequence_all_fn():
    # GT exists, but zero tracks
    gt = [GroundTruthAnnotation(1, 1, 10.0, 10.0, 20.0, 40.0, 1.0, 1, 1.0)]
    metrics = evaluate_tracking_sequence([], gt)
    assert metrics.total_gt == 1
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 0
    assert metrics.mota == 0.0
    assert metrics.idf1 == 0.0
    assert metrics.recall == 0.0


def test_evaluate_tracking_sequence_negative_mota():
    # 1 GT, 5 spurious false positive tracks -> MOTA = 1 - (1 FN + 5 FP)/1 = -5.0
    gt = [GroundTruthAnnotation(1, 1, 10.0, 10.0, 20.0, 40.0, 1.0, 1, 1.0)]
    tracks = [
        TrackObservation(i, 1, float(i * 100), 50.0, 20.0, 40.0, 0.9)
        for i in range(1, 6)
    ]
    metrics = evaluate_tracking_sequence(tracks, gt)
    assert metrics.total_gt == 1
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 5
    assert metrics.mota == -5.0

