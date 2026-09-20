"""Integration tests for Phase 12 Full-System Evaluation pipeline."""

import os
from pathlib import Path
import cv2
import pytest

from people_analytics.dataset.mot_sequence import MOT17Sequence
from people_analytics.detection.models import DetectionConfig
from people_analytics.detection.yolo_detector import YOLOv8PersonDetector
from people_analytics.evaluation import (
    FullPipelineEvaluator,
    generate_conservation_audit_log,
    generate_markdown_report,
    get_mot17_02_evaluation_config,
    get_mot17_09_evaluation_config,
    render_visual_artifacts,
    save_machine_readable_results,
    verify_reproducibility,
)
from people_analytics.footpoint.extractor import BottomCenterFootpointExtractor
from people_analytics.tracking.byte_tracker import ByteTracker
from people_analytics.tracking.models import TrackerConfig


LOCAL_MOT17_09 = Path("MOT17/MOT17/train/MOT17-09-FRCNN")
LOCAL_MOT17_02 = Path("MOT17/MOT17/train/MOT17-02-FRCNN")


@pytest.mark.skipif(not LOCAL_MOT17_09.is_dir(), reason="MOT17-09-FRCNN sequence directory not available")
def test_full_pipeline_evaluator_mot17_09_sample(tmp_path: Path):
    """Execute FullPipelineEvaluator on sample frames of MOT17-09 and verify all 7 invariants."""
    seq = MOT17Sequence(LOCAL_MOT17_09)
    config = get_mot17_09_evaluation_config()

    detector = YOLOv8PersonDetector(DetectionConfig(confidence_threshold=0.4))
    tracker = ByteTracker(TrackerConfig(high_threshold=0.4, min_hits=2))
    footpoint_extractor = BottomCenterFootpointExtractor()

    evaluator = FullPipelineEvaluator(
        detector=detector,
        tracker=tracker,
        footpoint_extractor=footpoint_extractor,
        config=config,
    )

    # Process 5 frames for fast integration verification
    report = evaluator.evaluate_sequence(seq, max_frames=5, evaluate_ground_truth=True)

    assert report.sequence_name == "MOT17-09-FRCNN"
    assert report.frames_processed == 5
    assert report.resolution == (1920, 1080)
    assert report.fps == 30.0

    # Invariants Audit (AC-03 to AC-07)
    assert report.conservation_audit.all_passed is True
    assert report.conservation_audit.projection_conservation.passed is True
    assert report.conservation_audit.image_heatmap_conservation.passed is True
    assert report.conservation_audit.ground_heatmap_conservation.passed is True
    assert report.conservation_audit.image_zone_conservation.passed is True
    assert report.conservation_audit.ground_zone_conservation.passed is True
    assert report.conservation_audit.image_dwell_conservation.passed is True
    assert report.conservation_audit.ground_dwell_conservation.passed is True

    # Check metrics existence
    assert report.detection_metrics is not None
    assert report.detection_metrics.true_positives >= 0
    assert report.tracking_metrics is not None
    assert report.trajectory_stats.total_observations > 0

    # Check cross-space comparison
    assert report.cross_space_comparison is not None
    assert 0.0 <= report.cross_space_comparison.agreement_ratio <= 1.0

    # Check artifact generation
    json_path = tmp_path / "evaluation_results_mot17_09.json"
    save_machine_readable_results(report, json_path)
    assert json_path.is_file()

    audit_log = generate_conservation_audit_log(report)
    assert "ALL PASSED" in audit_log

    md_path = tmp_path / "PHASE_12_EVALUATION_REPORT.md"
    md_content = generate_markdown_report([report], md_path)
    assert md_path.is_file()
    assert "AC-03" in md_content


@pytest.mark.skipif(not LOCAL_MOT17_02.is_dir(), reason="MOT17-02-FRCNN sequence directory not available")
def test_full_pipeline_evaluator_mot17_02_sample(tmp_path: Path):
    """Execute FullPipelineEvaluator on sample frames of MOT17-02 and verify all 7 invariants."""
    seq = MOT17Sequence(LOCAL_MOT17_02)
    config = get_mot17_02_evaluation_config()

    detector = YOLOv8PersonDetector(DetectionConfig(confidence_threshold=0.4))
    tracker = ByteTracker(TrackerConfig(high_threshold=0.4, min_hits=2))
    footpoint_extractor = BottomCenterFootpointExtractor()

    evaluator = FullPipelineEvaluator(
        detector=detector,
        tracker=tracker,
        footpoint_extractor=footpoint_extractor,
        config=config,
    )

    # Process 5 frames on secondary sequence
    report = evaluator.evaluate_sequence(seq, max_frames=5, evaluate_ground_truth=True)

    assert report.sequence_name == "MOT17-02-FRCNN"
    assert report.frames_processed == 5
    assert report.conservation_audit.all_passed is True


@pytest.mark.skipif(not LOCAL_MOT17_09.is_dir(), reason="MOT17-09-FRCNN sequence directory not available")
def test_full_pipeline_reproducibility_real_frames():
    """Verify consecutive executions on real video frames pass AC-08 reproducibility."""
    seq = MOT17Sequence(LOCAL_MOT17_09)
    config = get_mot17_09_evaluation_config()

    det_cfg = DetectionConfig(confidence_threshold=0.4)
    trk_cfg = TrackerConfig(high_threshold=0.4, min_hits=2)
    extractor = BottomCenterFootpointExtractor()

    # Run 1
    eval1 = FullPipelineEvaluator(YOLOv8PersonDetector(det_cfg), ByteTracker(trk_cfg), extractor, config)
    report1 = eval1.evaluate_sequence(seq, max_frames=5, evaluate_ground_truth=True)

    # Run 2
    eval2 = FullPipelineEvaluator(YOLOv8PersonDetector(det_cfg), ByteTracker(trk_cfg), extractor, config)
    report2 = eval2.evaluate_sequence(seq, max_frames=5, evaluate_ground_truth=True)

    result = verify_reproducibility(report1, report2, abs_tol=1e-5)
    assert result.is_reproducible is True
    assert len(result.differences) == 0


@pytest.mark.skipif(
    not os.environ.get("RUN_FULL_SEQUENCE"),
    reason="Full-sequence evaluation gated by RUN_FULL_SEQUENCE=1",
)
def test_full_525_frames_mot17_09():
    """Execute complete 525 frames of primary sequence MOT17-09-FRCNN (AC-01)."""
    seq = MOT17Sequence(LOCAL_MOT17_09)
    config = get_mot17_09_evaluation_config()
    evaluator = FullPipelineEvaluator(
        detector=YOLOv8PersonDetector(DetectionConfig(confidence_threshold=0.4)),
        tracker=ByteTracker(TrackerConfig(high_threshold=0.4, min_hits=2)),
        footpoint_extractor=BottomCenterFootpointExtractor(),
        config=config,
    )
    report = evaluator.evaluate_sequence(seq, max_frames=None, evaluate_ground_truth=True)
    assert report.frames_processed == 525
    assert report.conservation_audit.all_passed is True


@pytest.mark.skipif(
    not os.environ.get("RUN_FULL_SEQUENCE"),
    reason="Full-sequence evaluation gated by RUN_FULL_SEQUENCE=1",
)
def test_full_600_frames_mot17_02():
    """Execute complete 600 frames of secondary sequence MOT17-02-FRCNN (AC-02)."""
    seq = MOT17Sequence(LOCAL_MOT17_02)
    config = get_mot17_02_evaluation_config()
    evaluator = FullPipelineEvaluator(
        detector=YOLOv8PersonDetector(DetectionConfig(confidence_threshold=0.4)),
        tracker=ByteTracker(TrackerConfig(high_threshold=0.4, min_hits=2)),
        footpoint_extractor=BottomCenterFootpointExtractor(),
        config=config,
    )
    report = evaluator.evaluate_sequence(seq, max_frames=None, evaluate_ground_truth=True)
    assert report.frames_processed == 600
    assert report.conservation_audit.all_passed is True
