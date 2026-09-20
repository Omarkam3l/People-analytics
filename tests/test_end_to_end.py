"""End-to-end integration and engineering audit tests for Phase 9."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pytest

from people_analytics.dataset.mot_sequence import MOT17Sequence
from people_analytics.detection.base import BasePersonDetector
from people_analytics.detection.models import DetectionConfig, PersonDetection
from people_analytics.detection.yolo_detector import YOLOv8PersonDetector
from people_analytics.dwell.models import DwellConfig
from people_analytics.footpoint.extractor import BottomCenterFootpointExtractor
from people_analytics.footpoint.models import FootpointObservation
from people_analytics.heatmap.models import HeatmapConfig
from people_analytics.pipeline import (
    EndToEndReport,
    PipelineRunner,
    draw_pipeline_diagnostics,
)
from people_analytics.tracking.byte_tracker import ByteTracker
from people_analytics.tracking.models import TrackerConfig, TrackObservation
from people_analytics.zone.models import Zone, ZoneMembership


class MockDetector(BasePersonDetector):
    """Deterministic mock detector generating predictable person detections."""

    def __init__(self, detections_by_frame: dict[int, list[PersonDetection]]):
        super().__init__()
        self._dets = detections_by_frame

    def detect(self, image: np.ndarray, frame_index: int = 1) -> List[PersonDetection]:
        return self._dets.get(frame_index, [])

    def detect_frame(self, image_path: Path, frame_index: int = 1) -> List[PersonDetection]:
        return self._dets.get(frame_index, [])


def _create_test_zones() -> list[Zone]:
    z1 = Zone("Zone_Center", ((200.0, 200.0), (600.0, 200.0), (600.0, 600.0), (200.0, 600.0)))
    z2 = Zone("Zone_Corner", ((0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)))
    return [z1, z2]


# ====================================================================
# Invariant & Synthetic Pipeline Integration Tests
# ====================================================================

def test_end_to_end_mock_pipeline():
    """Verify complete end-to-end flow and all 7 integration invariants on mock data."""
    # Create 5 frames of detections for 2 moving targets:
    # Target 1 walks through Zone_Corner and Zone_Center
    # Target 2 walks in Zone_Center only
    mock_dets = {
        1: [
            PersonDetection(frame_index=1, bb_left=100.0, bb_top=100.0, bb_width=50.0, bb_height=100.0, confidence=0.9),
            PersonDetection(frame_index=1, bb_left=300.0, bb_top=300.0, bb_width=60.0, bb_height=120.0, confidence=0.9),
        ],
        2: [
            PersonDetection(frame_index=2, bb_left=110.0, bb_top=110.0, bb_width=50.0, bb_height=100.0, confidence=0.9),
            PersonDetection(frame_index=2, bb_left=310.0, bb_top=310.0, bb_width=60.0, bb_height=120.0, confidence=0.9),
        ],
        3: [
            PersonDetection(frame_index=3, bb_left=120.0, bb_top=120.0, bb_width=50.0, bb_height=100.0, confidence=0.9),
            PersonDetection(frame_index=3, bb_left=320.0, bb_top=320.0, bb_width=60.0, bb_height=120.0, confidence=0.9),
        ],
        4: [
            PersonDetection(frame_index=4, bb_left=130.0, bb_top=130.0, bb_width=50.0, bb_height=100.0, confidence=0.9),
            PersonDetection(frame_index=4, bb_left=330.0, bb_top=330.0, bb_width=60.0, bb_height=120.0, confidence=0.9),
        ],
        5: [
            PersonDetection(frame_index=5, bb_left=140.0, bb_top=140.0, bb_width=50.0, bb_height=100.0, confidence=0.9),
            PersonDetection(frame_index=5, bb_left=340.0, bb_top=340.0, bb_width=60.0, bb_height=120.0, confidence=0.9),
        ],
    }

    detector = MockDetector(mock_dets)
    tracker = ByteTracker(TrackerConfig(high_threshold=0.5, min_hits=1))
    extractor = BottomCenterFootpointExtractor()
    hm_config = HeatmapConfig(image_width=1920, image_height=1080, cell_size=50)
    zones = _create_test_zones()
    dwell_config = DwellConfig(fps=30.0, max_gap_frames=0)

    runner = PipelineRunner(
        detector=detector,
        tracker=tracker,
        footpoint_extractor=extractor,
        heatmap_config=hm_config,
        zones=zones,
        dwell_config=dwell_config,
    )

    seq_path = Path("MOT17/MOT17/train/MOT17-09-FRCNN")
    if not seq_path.is_dir():
        pytest.skip("MOT17-09-FRCNN sequence not found.")

    seq = MOT17Sequence(seq_path)
    report = runner.process_sequence(seq, max_frames=5, evaluate_ground_truth=False)

    # Invariant Verification
    assert report.all_invariants_passed is True
    assert len(report.invariants) == 7

    # Spatial Diagnostics
    assert report.spatial.total_detections == 10
    assert report.spatial.total_track_observations == 10
    assert report.spatial.total_footpoints == 10
    assert report.spatial.total_trajectories == 2

    # Heatmap Diagnostics
    assert report.heatmap.total_accumulated == 10
    assert report.heatmap.out_of_bounds_points == 0
    assert report.heatmap.invalid_points == 0
    assert report.heatmap.occupied_cells > 0

    # Zone & Dwell Diagnostics
    assert report.zone.total_memberships == 10
    assert report.dwell.total_visits >= 2


def test_draw_pipeline_diagnostics_does_not_mutate():
    """Verify diagnostic visualization does not mutate input canvas and renders all layers."""
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas_copy = canvas.copy()

    tracks = [
        TrackObservation(1, 10, 100.0, 100.0, 50.0, 100.0, 0.9),
    ]
    footpoints = [
        FootpointObservation(1, 10, 125.0, 200.0, 0.9),
    ]
    zones = _create_test_zones()
    memberships = [
        ZoneMembership(1, 10, 125.0, 200.0, ("Zone_Corner",)),
    ]

    output = draw_pipeline_diagnostics(
        image=canvas,
        tracks=tracks,
        footpoints=footpoints,
        zones=zones,
        memberships=memberships,
        frame_index=10,
        throughput_fps=28.5,
    )

    assert output.shape == canvas.shape
    assert output.dtype == canvas.dtype
    assert np.array_equal(canvas, canvas_copy)
    assert not np.array_equal(output, canvas)


# ====================================================================
# Real MOT17-09-FRCNN End-to-End Test
# ====================================================================

def test_real_mot17_09_end_to_end_sample():
    """Execute end-to-end pipeline with YOLOv8 on real MOT17-09 frames and verify invariants."""
    seq_path = Path("MOT17/MOT17/train/MOT17-09-FRCNN")
    if not seq_path.is_dir():
        pytest.skip("MOT17-09-FRCNN sequence directory not available.")

    seq = MOT17Sequence(seq_path)
    width = seq.info.im_width
    height = seq.info.im_height
    fps = seq.info.frame_rate

    detector = YOLOv8PersonDetector(DetectionConfig(confidence_threshold=0.4))
    tracker = ByteTracker(TrackerConfig(high_threshold=0.4, min_hits=2))
    extractor = BottomCenterFootpointExtractor()
    hm_config = HeatmapConfig(image_width=width, image_height=height, cell_size=50)

    # Simple central sidewalk test zone
    zone = Zone(
        zone_id="Sidewalk",
        name="Sidewalk",
        vertices=(
            (0.1 * width, 0.3 * height),
            (0.9 * width, 0.3 * height),
            (0.9 * width, 0.9 * height),
            (0.1 * width, 0.9 * height),
        ),
    )
    dwell_config = DwellConfig(fps=fps, max_gap_frames=0)

    runner = PipelineRunner(
        detector=detector,
        tracker=tracker,
        footpoint_extractor=extractor,
        heatmap_config=hm_config,
        zones=[zone],
        dwell_config=dwell_config,
    )

    # Process 10 frames of MOT17-09
    report = runner.process_sequence(seq, max_frames=10, evaluate_ground_truth=True)

    assert report.sequence_name == "MOT17-09-FRCNN"
    assert report.frames_processed == 10
    assert report.all_invariants_passed is True

    # Check metrics existence
    assert report.detection_metrics is not None
    assert report.detection_metrics.precision >= 0.0
    assert report.tracking_metrics is not None
    assert report.timing.total_ms > 0.0
    assert report.timing.fps > 0.0

    # Check spatial conservation equations
    assert report.spatial.total_track_observations == report.spatial.total_footpoints
    assert report.spatial.total_footpoints == (
        report.heatmap.total_accumulated + report.heatmap.out_of_bounds_points + report.heatmap.invalid_points
    )
    assert report.spatial.total_footpoints == (
        report.zone.inside_at_least_one_zone + report.zone.outside_all_zones
    )
    assert report.zone.total_memberships == report.spatial.total_footpoints
    assert report.zone.inside_at_least_one_zone == (
        report.zone.single_zone_observations + report.zone.multi_zone_observations
    )


def test_conservation_and_zone_diagnostics_semantics():
    """Verify exact conservation equations and multi-zone overlap diagnostics."""
    # 3 frames, 3 tracks:
    # Point A at (50, 50) -> in z1 only
    # Point B at (150, 150) -> in z1 and z2 (overlapping)
    # Point C at (500, 500) -> outside all zones
    mock_dets = {
        1: [
            PersonDetection(frame_index=1, bb_left=40.0, bb_top=20.0, bb_width=20.0, bb_height=30.0, confidence=0.9),    # fp=(50, 50)
            PersonDetection(frame_index=1, bb_left=140.0, bb_top=120.0, bb_width=20.0, bb_height=30.0, confidence=0.9),  # fp=(150, 150)
            PersonDetection(frame_index=1, bb_left=490.0, bb_top=470.0, bb_width=20.0, bb_height=30.0, confidence=0.9),  # fp=(500, 500)
        ],
    }

    z1 = Zone("Zone_1", ((0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)))
    z2 = Zone("Zone_2", ((100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)))

    detector = MockDetector(mock_dets)
    tracker = ByteTracker(TrackerConfig(high_threshold=0.5, min_hits=1))
    extractor = BottomCenterFootpointExtractor()
    hm_config = HeatmapConfig(image_width=1000, image_height=1000, cell_size=50)
    dwell_config = DwellConfig(fps=30.0, max_gap_frames=0)

    runner = PipelineRunner(
        detector=detector,
        tracker=tracker,
        footpoint_extractor=extractor,
        heatmap_config=hm_config,
        zones=[z1, z2],
        dwell_config=dwell_config,
    )

    seq_path = Path("MOT17/MOT17/train/MOT17-09-FRCNN")
    seq = MOT17Sequence(seq_path)
    report = runner.process_sequence(seq, max_frames=1, evaluate_ground_truth=False)

    # Spatial conservation 1: Footpoints == Heatmap accumulated + OOB + Invalid
    assert report.spatial.total_footpoints == (
        report.heatmap.total_accumulated + report.heatmap.out_of_bounds_points + report.heatmap.invalid_points
    )
    assert report.spatial.total_footpoints == 3
    assert report.heatmap.total_accumulated == 3
    assert report.heatmap.out_of_bounds_points == 0
    assert report.heatmap.invalid_points == 0

    # Spatial conservation 2: Footpoints == inside_at_least_one + outside_all
    assert report.spatial.total_footpoints == (
        report.zone.inside_at_least_one_zone + report.zone.outside_all_zones
    )

    # Detailed breakdown:
    # 1 single-zone (Point A), 1 multi-zone (Point B), 1 outside all (Point C)
    assert report.zone.total_memberships == 3
    assert report.zone.single_zone_observations == 1
    assert report.zone.multi_zone_observations == 1
    assert report.zone.outside_all_zones == 1
    assert report.zone.inside_at_least_one_zone == 2
    assert report.zone.in_zone_observations == 2
    assert report.zone.outside_observations == 1

    # Invariant naming check
    inv_names = [inv.invariant_name for inv in report.invariants]
    assert "INV-2: Track-to-Footpoint Propagation & Conservation" in inv_names
    assert "INV-5: Footpoint-to-Zone Eligibility & Propagation" in inv_names
    assert report.all_invariants_passed is True
