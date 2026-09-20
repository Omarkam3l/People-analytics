"""Comprehensive unit tests for Phase 12: Full-System Evaluation logic."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from people_analytics.detection.models import PersonDetection
from people_analytics.dwell.models import DwellConfig, ZoneVisit
from people_analytics.evaluation import (
    ConservationCheck,
    DetectionMetrics,
    EvaluationFrameState,
    FullConservationAudit,
    FullPipelineEvaluator,
    FullSequenceEvaluationReport,
    GroundProjectionDiagnostics,
    SceneEvaluationConfig,
    TrackingMetrics,
    TrajectorySummaryStats,
    generate_conservation_audit_log,
    generate_markdown_report,
    get_mot17_09_evaluation_config,
    get_report_markdown_path,
    render_visual_artifacts,
    save_machine_readable_results,
    verify_reproducibility,
)
from people_analytics.footpoint.extractor import BottomCenterFootpointExtractor
from people_analytics.footpoint.models import FootpointObservation
from people_analytics.ground.models import CoordinateFrame, GroundPoint
from people_analytics.ground_analytics.comparison import GroundComparisonReport
from people_analytics.ground_analytics.models import GroundObservation, GroundZone
from people_analytics.pipeline.models import (
    DwellDiagnostics,
    HeatmapDiagnostics,
    SpatialDiagnostics,
    StageTiming,
    ZoneDiagnostics,
)
from people_analytics.tracking.models import TrackObservation
from people_analytics.trajectory.models import Trajectory
from people_analytics.zone.models import Zone, ZoneMembership


# ====================================================================
# Helpers & Fixtures
# ====================================================================

def _create_sample_report(
    seq_name: str = "MOT17-09-FRCNN",
    tp: int = 100,
    mota: float = 0.65432,
    total_ms: float = 1200.0,
    rss_mb: float = 45.2,
    all_invariants_pass: bool = True,
) -> FullSequenceEvaluationReport:
    """Helper creating a fully populated sample evaluation report."""
    traj_stats = TrajectorySummaryStats(
        total_trajectories=5,
        total_observations=120,
        mean_observations_per_track=24.0,
        trajectories_with_gaps=2,
        total_gaps=3,
        min_gap_frames=1,
        max_gap_frames=4,
        mean_gap_frames=2.33,
        median_gap_frames=2.0,
        min_lifespan_frames=10,
        max_lifespan_frames=40,
        mean_lifespan_frames=26.0,
        median_lifespan_frames=25.0,
        min_lifespan_seconds=0.33,
        max_lifespan_seconds=1.33,
        mean_lifespan_seconds=0.87,
        median_lifespan_seconds=0.83,
        frame_coverage_ratio=0.923,
    )
    img_spatial = SpatialDiagnostics(
        total_detections=150,
        total_track_observations=120,
        total_footpoints=120,
        valid_footpoints=120,
        invalid_footpoints=0,
        out_of_bounds_footpoints=0,
        total_trajectories=5,
        trajectories_with_gaps=2,
        total_gaps=3,
    )
    img_heatmap = HeatmapDiagnostics(
        total_accumulated=120,
        grid_shape=(22, 39),
        occupied_cells=45,
        occupancy_ratio=45 / (22 * 39),
        max_cell_count=8.0,
        out_of_bounds_points=0,
        invalid_points=0,
    )
    img_zone = ZoneDiagnostics(
        total_memberships=120,
        inside_at_least_one_zone=80,
        outside_all_zones=40,
        single_zone_observations=70,
        multi_zone_observations=10,
        unique_visitors_per_zone={"Zone_Central": 4, "Zone_Left": 2},
    )
    img_dwell = DwellDiagnostics(
        total_visits=6,
        unique_visitors=4,
        total_dwell_seconds=18.5,
        average_dwell_seconds=3.08,
        max_dwell_seconds=6.2,
        visits_by_zone={"Zone_Central": 4, "Zone_Left": 2},
    )
    grd_proj = GroundProjectionDiagnostics(
        total_projected=120,
        valid_in_roi=100,
        valid_extrapolated=18,
        invalid_projections=2,
        analytics_accepted=118,
        analytics_excluded=2,
        error_codes={"PROJECTIVE_DENOMINATOR_TOO_SMALL": 2},
    )
    cmp_rep = GroundComparisonReport(
        image_visits_total=6,
        ground_visits_total=6,
        image_unique_visitors=4,
        ground_unique_visitors=4,
        image_dwell_seconds_total=18.5,
        ground_dwell_seconds_total=17.8,
        visits_by_zone_image={"Zone_Central": 4, "Zone_Left": 2},
        visits_by_zone_ground={"Zone_Ground_Central": 4, "Zone_Ground_Left": 2},
        membership_agreement_count=110,
        membership_total_evaluated=120,
        agreement_ratio=110 / 120,
    )

    pass_status = all_invariants_pass
    audit = FullConservationAudit(
        projection_conservation=ConservationCheck("AC-03", pass_status, 120, 120 if pass_status else 118, "Projection Conservation"),
        image_heatmap_conservation=ConservationCheck("AC-04", pass_status, 120, 120, "Image Heatmap Conservation"),
        ground_heatmap_conservation=ConservationCheck("AC-05", pass_status, 120, 120, "Ground Heatmap Conservation"),
        image_zone_conservation=ConservationCheck("AC-06a", pass_status, 120, 120, "Image Zone Conservation"),
        ground_zone_conservation=ConservationCheck("AC-06b", pass_status, 120, 120, "Ground Zone Conservation"),
        image_dwell_conservation=ConservationCheck("AC-07a", pass_status, 90, 90, "Image Dwell Overlap Conservation"),
        ground_dwell_conservation=ConservationCheck("AC-07b", pass_status, 90, 90, "Ground Dwell Overlap Conservation"),
    )

    timing = StageTiming(
        detection_ms=total_ms * 0.4,
        tracking_ms=total_ms * 0.2,
        footpoint_ms=total_ms * 0.05,
        trajectory_ms=total_ms * 0.1,
        heatmap_ms=total_ms * 0.1,
        zone_ms=total_ms * 0.05,
        dwell_ms=total_ms * 0.1,
        total_ms=total_ms,
        fps=120 / (total_ms / 1000.0),
    )

    return FullSequenceEvaluationReport(
        sequence_name=seq_name,
        frames_processed=50,
        total_sequence_frames=525,
        fps=30.0,
        resolution=(1920, 1080),
        detection_metrics=DetectionMetrics(tp, 10, 15, 0.909, 0.869, 0.889),
        tracking_metrics=TrackingMetrics(115, tp, 10, 15, 2, mota, 0.725, 0.909, 0.869),
        trajectory_stats=traj_stats,
        image_spatial=img_spatial,
        image_heatmap=img_heatmap,
        image_zone=img_zone,
        image_dwell=img_dwell,
        ground_projection=grd_proj,
        ground_heatmap_accumulated=115,
        ground_heatmap_out_of_bounds=3,
        ground_heatmap_invalid=2,
        ground_heatmap_extrap_rejected=0,
        ground_heatmap_occupied_cells=40,
        ground_heatmap_max_cell_count=7.0,
        ground_zone_total_memberships=120,
        ground_zone_inside=75,
        ground_zone_outside=45,
        ground_zone_single=65,
        ground_zone_multi=10,
        ground_dwell_total_visits=6,
        ground_dwell_unique_visitors=4,
        ground_dwell_total_seconds=17.8,
        ground_dwell_average_seconds=2.97,
        ground_dwell_visits_by_zone={"Zone_Ground_Central": 4, "Zone_Ground_Left": 2},
        cross_space_comparison=cmp_rep,
        conservation_audit=audit,
        timing=timing,
        peak_rss_mb=rss_mb,
    )


# ====================================================================
# 1. Trajectory Summary Statistics Tests
# ====================================================================

def test_trajectory_summary_statistics_computation():
    """Verify trajectory statistics calculation on deterministic trajectories."""
    # Track 1: frames 1, 2, 5 (gap of 2 frames between 2 and 5) -> lifespan = 5 - 1 + 1 = 5
    t1 = Trajectory(
        track_id=1,
        points=(
            FootpointObservation(1, 1, 10.0, 10.0),
            FootpointObservation(1, 2, 11.0, 11.0),
            FootpointObservation(1, 5, 12.0, 12.0),
        ),
    )
    # Track 2: frames 3, 4, 5 (no gaps) -> lifespan = 5 - 3 + 1 = 3
    t2 = Trajectory(
        track_id=2,
        points=(
            FootpointObservation(2, 3, 20.0, 20.0),
            FootpointObservation(2, 4, 21.0, 21.0),
            FootpointObservation(2, 5, 22.0, 22.0),
        ),
    )

    evaluator = FullPipelineEvaluator(MagicMock(), MagicMock(), MagicMock(), get_mot17_09_evaluation_config())
    stats = evaluator._compute_trajectory_stats({1: t1, 2: t2}, fps=10.0)

    assert stats.total_trajectories == 2
    assert stats.total_observations == 6
    assert stats.mean_observations_per_track == 3.0
    assert stats.trajectories_with_gaps == 1
    assert stats.total_gaps == 1
    assert stats.min_gap_frames == 2
    assert stats.max_gap_frames == 2
    assert stats.mean_gap_frames == 2.0
    assert stats.median_gap_frames == 2.0

    # Lifespans: 5 frames (0.5s) and 3 frames (0.3s)
    assert stats.min_lifespan_frames == 3
    assert stats.max_lifespan_frames == 5
    assert stats.mean_lifespan_frames == 4.0
    assert stats.median_lifespan_frames == 4.0
    assert np.isclose(stats.min_lifespan_seconds, 0.3)
    assert np.isclose(stats.max_lifespan_seconds, 0.5)

    # Frame coverage: 6 observations / (5 + 3 = 8 possible frames) = 0.75
    assert np.isclose(stats.frame_coverage_ratio, 0.75)


def test_trajectory_summary_statistics_empty():
    """Verify trajectory statistics handle empty input gracefully."""
    evaluator = FullPipelineEvaluator(MagicMock(), MagicMock(), MagicMock(), get_mot17_09_evaluation_config())
    stats = evaluator._compute_trajectory_stats({}, fps=30.0)

    assert stats.total_trajectories == 0
    assert stats.total_observations == 0
    assert stats.frame_coverage_ratio == 0.0


# ====================================================================
# 2. Conservation Invariants Audit Tests (AC-03 to AC-07)
# ====================================================================

def test_full_conservation_audit_all_passed():
    """Verify FullConservationAudit reports all_passed=True when all 7 equations hold."""
    report = _create_sample_report(all_invariants_pass=True)
    assert report.conservation_audit.all_passed is True
    assert len(report.conservation_audit.checks) == 7
    for c in report.conservation_audit.checks:
        assert c.passed is True
        assert c.left_value == c.right_value


def test_full_conservation_audit_detects_failure():
    """Verify FullConservationAudit reports all_passed=False when any equation fails."""
    report = _create_sample_report(all_invariants_pass=False)
    assert report.conservation_audit.all_passed is False


# ====================================================================
# 3. Reproducibility Verification Tests (AC-08)
# ====================================================================

def test_reproducibility_identical_runs():
    """Verify identical evaluation reports pass AC-08 reproducibility check."""
    r1 = _create_sample_report(tp=100, mota=0.65432, total_ms=1000.0, rss_mb=50.0)
    # Consecutive run with same metrics but varying runtime and memory
    r2 = _create_sample_report(tp=100, mota=0.65432, total_ms=1350.0, rss_mb=52.5)

    result = verify_reproducibility(r1, r2, abs_tol=1e-5)
    assert result.is_reproducible is True
    assert len(result.differences) == 0
    assert result.total_checks > 20


def test_reproducibility_detects_discrete_mismatch():
    """Verify discrete count mismatch fails AC-08."""
    r1 = _create_sample_report(tp=100)
    r2 = _create_sample_report(tp=101)  # 1 count mismatch

    result = verify_reproducibility(r1, r2, abs_tol=1e-5)
    assert result.is_reproducible is False
    assert any(d.difference_type == "DISCRETE_MISMATCH" for d in result.differences)


def test_reproducibility_floating_point_tolerance():
    """Verify float tolerance honors abs_tol <= 1e-5."""
    # Within tolerance (1e-6 diff)
    r1 = _create_sample_report(mota=0.654321)
    r2 = _create_sample_report(mota=0.654322)
    assert verify_reproducibility(r1, r2, abs_tol=1e-5).is_reproducible is True

    # Exceeding tolerance (1e-4 diff)
    r3 = _create_sample_report(mota=0.654421)
    res_fail = verify_reproducibility(r1, r3, abs_tol=1e-5)
    assert res_fail.is_reproducible is False
    assert any(d.difference_type == "FLOAT_TOLERANCE_EXCEEDED" for d in res_fail.differences)


def test_reproducibility_detects_invariant_failure():
    """Verify invariant failure divergence is caught."""
    r1 = _create_sample_report(all_invariants_pass=True)
    r2 = _create_sample_report(all_invariants_pass=False)

    result = verify_reproducibility(r1, r2, abs_tol=1e-5)
    assert result.is_reproducible is False
    assert any(d.difference_type == "INVARIANT_MISMATCH" for d in result.differences)


# ====================================================================
# 4. Serialization & Report Generation Tests
# ====================================================================

def test_report_json_serialization(tmp_path: Path):
    """Verify JSON schema serialization round-trip."""
    report = _create_sample_report()
    json_path = tmp_path / "evaluation_results_mot17_09.json"

    save_machine_readable_results(report, json_path)
    assert json_path.is_file()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["sequence_name"] == "MOT17-09-FRCNN"
    assert data["frames_processed"] == 50
    assert data["detection_metrics"]["true_positives"] == 100
    assert data["tracking_metrics"]["mota"] == pytest.approx(0.65432)
    assert data["conservation_audit"]["all_passed"] is True
    assert len(data["conservation_audit"]["checks"]) == 7


def test_markdown_report_generation(tmp_path: Path):
    """Verify markdown evaluation report contains all required sections and limitations."""
    r1 = _create_sample_report("MOT17-09-FRCNN")
    r2 = _create_sample_report("MOT17-02-FRCNN")

    md_path = tmp_path / "PHASE_12_EVALUATION_REPORT.md"
    content = generate_markdown_report([r1, r2], md_path)

    assert md_path.is_file()
    assert "Phase 12: Full-System Evaluation Report" in content
    assert "MOT17-09-FRCNN" in content
    assert "MOT17-02-FRCNN" in content
    assert "Detection Evaluation (Image Plane)" in content
    assert "Multi-Object Tracking Evaluation" in content
    assert "Mathematical Conservation Audit (AC-03 to AC-07)" in content
    assert "ARBITRARY_PLANAR" in content
    assert "Known Limitations & Verification Scope" in content


def test_conservation_audit_log_format():
    """Verify console/file conservation audit log formatting."""
    report = _create_sample_report()
    log = generate_conservation_audit_log(report)

    assert "CONSERVATION AUDIT LOG: MOT17-09-FRCNN" in log
    assert "AC-03" in log
    assert "AC-04" in log
    assert "AC-05" in log
    assert "Overall Audit Status: ALL PASSED" in log


# ====================================================================
# 5. Targeted Hardening Tests (Finding 1 & Finding 2)
# ====================================================================

def test_markdown_report_path_resolution(tmp_path: Path):
    """Verify markdown report path resolution for single sequences and consolidated run."""
    p_09 = get_report_markdown_path(tmp_path, "MOT17-09-FRCNN")
    p_02 = get_report_markdown_path(tmp_path, "MOT17-02-FRCNN")
    p_all = get_report_markdown_path(tmp_path, "all")

    assert p_09 == tmp_path / "PHASE_12_EVALUATION_REPORT_MOT17_09.md"
    assert p_02 == tmp_path / "PHASE_12_EVALUATION_REPORT_MOT17_02.md"
    assert p_all == tmp_path / "PHASE_12_EVALUATION_REPORT.md"


def test_markdown_report_no_overwrite_between_sequences(tmp_path: Path):
    """Proves running sequence A does not overwrite sequence B's markdown report."""
    r1 = _create_sample_report("MOT17-09-FRCNN")
    r2 = _create_sample_report("MOT17-02-FRCNN")

    path_09 = get_report_markdown_path(tmp_path, "MOT17-09-FRCNN")
    path_02 = get_report_markdown_path(tmp_path, "MOT17-02-FRCNN")

    assert path_09 != path_02

    # Run / generate sequence 09
    generate_markdown_report([r1], path_09)
    assert path_09.is_file()
    content_09_before = path_09.read_text(encoding="utf-8")

    # Run / generate sequence 02
    generate_markdown_report([r2], path_02)
    assert path_02.is_file()

    # Verify sequence 09 report was NOT overwritten
    content_09_after = path_09.read_text(encoding="utf-8")
    assert content_09_before == content_09_after


def test_sequence_specific_reports_contain_only_corresponding_results(tmp_path: Path):
    """Proves sequence-specific reports contain only their corresponding sequence results."""
    r1 = _create_sample_report("MOT17-09-FRCNN")
    r2 = _create_sample_report("MOT17-02-FRCNN")

    path_09 = get_report_markdown_path(tmp_path, "MOT17-09-FRCNN")
    path_02 = get_report_markdown_path(tmp_path, "MOT17-02-FRCNN")

    content_09 = generate_markdown_report([r1], path_09)
    assert "MOT17-09-FRCNN" in content_09
    assert "MOT17-02-FRCNN" not in content_09

    content_02 = generate_markdown_report([r2], path_02)
    assert "MOT17-02-FRCNN" in content_02
    assert "MOT17-09-FRCNN" not in content_02


def test_sequence_all_produces_consolidated_report(tmp_path: Path):
    """Proves --sequence all produces the consolidated report covering all sequences."""
    r1 = _create_sample_report("MOT17-09-FRCNN")
    r2 = _create_sample_report("MOT17-02-FRCNN")

    path_all = get_report_markdown_path(tmp_path, "all")
    assert path_all.name == "PHASE_12_EVALUATION_REPORT.md"

    content_all = generate_markdown_report([r1, r2], path_all)
    assert path_all.is_file()
    assert "MOT17-09-FRCNN" in content_all
    assert "MOT17-02-FRCNN" in content_all


def test_evaluator_captures_active_frame_state():
    """Proves evaluator captures non-empty active frame state when observations exist."""
    det_mock = MagicMock()
    trk_mock = MagicMock()
    fp_mock = MagicMock()
    seq_mock = MagicMock()
    cfg = get_mot17_09_evaluation_config()

    # Frame 1 & 2 return detections and tracks
    det_mock.detect_frame.side_effect = lambda path, frame_index: [
        PersonDetection(frame_index=frame_index, bb_left=100.0, bb_top=100.0, bb_width=50.0, bb_height=100.0, confidence=0.9)
    ]
    trk_mock.update.side_effect = lambda dets, frame_index: [
        TrackObservation(track_id=1, frame_index=frame_index, bb_left=100.0, bb_top=100.0, bb_width=50.0, bb_height=100.0, confidence=0.9)
    ]
    fp_mock.extract_batch.side_effect = lambda trks: [
        FootpointObservation(track_id=t.track_id, frame_index=t.frame_index, x=125.0, y=200.0, confidence=0.9)
        for t in trks
    ]

    seq_mock.info.name = "MOT17-09-FRCNN"
    seq_mock.info.seq_length = 2
    seq_mock.get_frame_path.return_value = Path("dummy_frame.jpg")
    seq_mock.has_ground_truth.return_value = False

    evaluator = FullPipelineEvaluator(det_mock, trk_mock, fp_mock, cfg)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "is_file", lambda self: True)
        evaluator.evaluate_sequence(seq_mock, max_frames=2, evaluate_ground_truth=False)

    assert evaluator.last_frame_state is not None
    assert evaluator.last_frame_state.frame_index == 2
    assert len(evaluator.last_frame_tracks) == 1
    assert evaluator.last_frame_tracks[0].track_id == 1
    assert len(evaluator.last_frame_footpoints) == 1
    assert len(evaluator.last_frame_ground_obs) == 1
    assert evaluator.last_frame_ground_obs[0].is_valid is True
    assert len(evaluator.last_frame_img_memberships) == 1
    assert evaluator.last_ground_heatmap_data is not None


def test_visualization_renderer_receives_non_empty_state_and_associates_sequence(tmp_path: Path):
    """Proves visualization renderer receives non-empty state and associates artifact with sequence."""
    r_09 = _create_sample_report("MOT17-09-FRCNN")
    r_02 = _create_sample_report("MOT17-02-FRCNN")

    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracks = [TrackObservation(track_id=1, frame_index=50, bb_left=200.0, bb_top=200.0, bb_width=60.0, bb_height=120.0, confidence=0.95)]
    fps = [FootpointObservation(track_id=1, frame_index=50, x=230.0, y=320.0, confidence=0.95)]
    ground_obs = [
        GroundObservation(
            track_id=1,
            frame_index=50,
            x=10.5,
            y=25.3,
            frame=CoordinateFrame.ARBITRARY_PLANAR,
            is_valid=True,
            is_extrapolated=False,
        )
    ]
    img_zones = [Zone(zone_id="Z1", name="Zone 1", vertices=((100.0, 100.0), (400.0, 100.0), (400.0, 400.0), (100.0, 400.0)))]
    grd_zones = [GroundZone(zone_id="GZ1", name="Ground Zone 1", vertices=((5.0, 5.0), (30.0, 5.0), (30.0, 30.0), (5.0, 30.0)))]

    # MOT17-09
    comp_09, hm_09 = render_visual_artifacts(
        report=r_09,
        last_frame_bgr=img,
        tracks=tracks,
        footpoints=fps,
        ground_obs=ground_obs,
        image_zones=img_zones,
        ground_zones=grd_zones,
        ground_heatmap_data=None,
        output_dir=tmp_path,
    )

    assert comp_09.name == "mot17_09_frcnn_dual_view_composite.png"
    assert hm_09.name == "mot17_09_frcnn_ground_heatmap.png"
    assert comp_09.is_file() and comp_09.stat().st_size > 0
    assert hm_09.is_file() and hm_09.stat().st_size > 0

    # MOT17-02
    comp_02, hm_02 = render_visual_artifacts(
        report=r_02,
        last_frame_bgr=img,
        tracks=tracks,
        footpoints=fps,
        ground_obs=ground_obs,
        image_zones=img_zones,
        ground_zones=grd_zones,
        ground_heatmap_data=None,
        output_dir=tmp_path,
    )

    assert comp_02.name == "mot17_02_frcnn_dual_view_composite.png"
    assert hm_02.name == "mot17_02_frcnn_ground_heatmap.png"
    assert comp_02.is_file() and comp_02.stat().st_size > 0
    assert hm_02.is_file() and hm_02.stat().st_size > 0


def test_visualization_backward_compatibility_empty_state(tmp_path: Path):
    """Proves existing visualization functionality remains intact when observations are empty."""
    report = _create_sample_report("MOT17-09-FRCNN")
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    comp_p, hm_p = render_visual_artifacts(
        report=report,
        last_frame_bgr=img,
        tracks=[],
        footpoints=[],
        ground_obs=[],
        image_zones=[],
        ground_zones=[],
        ground_heatmap_data=None,
        output_dir=tmp_path,
    )

    assert comp_p.is_file()
    assert hm_p.is_file()
    assert comp_p.name == "mot17_09_frcnn_dual_view_composite.png"

