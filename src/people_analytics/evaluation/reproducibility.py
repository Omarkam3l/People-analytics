"""Reproducibility verification between consecutive evaluation runs."""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

from people_analytics.evaluation.models import FullSequenceEvaluationReport


@dataclass(frozen=True)
class ReproducibilityDifference:
    """Discrepancy recorded during reproducibility verification."""
    field_path: str
    run1_value: Any
    run2_value: Any
    difference_type: str  # "DISCRETE_MISMATCH", "FLOAT_TOLERANCE_EXCEEDED", "INVARIANT_MISMATCH"


@dataclass(frozen=True)
class ReproducibilityReport:
    """Outcome of comparing two consecutive evaluation runs under AC-08."""
    is_reproducible: bool
    sequence_name: str
    tolerance_used: float
    total_checks: int
    differences: Tuple[ReproducibilityDifference, ...] = ()


def verify_reproducibility(
    run1: FullSequenceEvaluationReport,
    run2: FullSequenceEvaluationReport,
    abs_tol: float = 1e-5,
) -> ReproducibilityReport:
    """Verify that two evaluation reports satisfy AC-08 reproducibility constraints.

    Requirements:
        - Discrete counts must match exactly (integer equality).
        - Invariant and conservation results must match exactly (boolean equality).
        - Floating-point aggregate metrics must match within abs_tol <= 1e-5.
        - Runtime, latency, FPS, and peak RSS memory are excluded from deterministic equality.

    Args:
        run1: First evaluation report.
        run2: Second evaluation report from an identical run.
        abs_tol: Maximum absolute tolerance for floating-point values (default: 1e-5).

    Returns:
        ReproducibilityReport documenting compliance and any recorded discrepancies.
    """
    diffs: List[ReproducibilityDifference] = []
    total_checks = 0

    def check_exact(path: str, val1: Any, val2: Any, diff_type: str = "DISCRETE_MISMATCH") -> None:
        nonlocal total_checks
        total_checks += 1
        if val1 != val2:
            diffs.append(ReproducibilityDifference(path, val1, val2, diff_type))

    def check_float(path: str, val1: Optional[float], val2: Optional[float]) -> None:
        nonlocal total_checks
        total_checks += 1
        if val1 is None and val2 is None:
            return
        if val1 is None or val2 is None:
            diffs.append(ReproducibilityDifference(path, val1, val2, "DISCRETE_MISMATCH"))
            return
        if not math.isclose(val1, val2, abs_tol=abs_tol):
            diffs.append(ReproducibilityDifference(path, val1, val2, "FLOAT_TOLERANCE_EXCEEDED"))

    # Sequence metadata
    check_exact("sequence_name", run1.sequence_name, run2.sequence_name)
    check_exact("frames_processed", run1.frames_processed, run2.frames_processed)
    check_exact("total_sequence_frames", run1.total_sequence_frames, run2.total_sequence_frames)
    check_exact("resolution", run1.resolution, run2.resolution)

    # Invariants (AC-03 to AC-07)
    check_exact(
        "conservation_audit.all_passed",
        run1.conservation_audit.all_passed,
        run2.conservation_audit.all_passed,
        "INVARIANT_MISMATCH",
    )
    for c1, c2 in zip(run1.conservation_audit.checks, run2.conservation_audit.checks):
        check_exact(f"conservation.{c1.name}.passed", c1.passed, c2.passed, "INVARIANT_MISMATCH")
        check_exact(f"conservation.{c1.name}.left", c1.left_value, c2.left_value)
        check_exact(f"conservation.{c1.name}.right", c1.right_value, c2.right_value)

    # Detection metrics
    if run1.detection_metrics and run2.detection_metrics:
        dm1, dm2 = run1.detection_metrics, run2.detection_metrics
        check_exact("detection.true_positives", dm1.true_positives, dm2.true_positives)
        check_exact("detection.false_positives", dm1.false_positives, dm2.false_positives)
        check_exact("detection.false_negatives", dm1.false_negatives, dm2.false_negatives)
        check_float("detection.precision", dm1.precision, dm2.precision)
        check_float("detection.recall", dm1.recall, dm2.recall)
        check_float("detection.f1_score", dm1.f1_score, dm2.f1_score)

    # Tracking metrics
    if run1.tracking_metrics and run2.tracking_metrics:
        tm1, tm2 = run1.tracking_metrics, run2.tracking_metrics
        check_exact("tracking.total_gt", tm1.total_gt, tm2.total_gt)
        check_exact("tracking.true_positives", tm1.true_positives, tm2.true_positives)
        check_exact("tracking.false_positives", tm1.false_positives, tm2.false_positives)
        check_exact("tracking.false_negatives", tm1.false_negatives, tm2.false_negatives)
        check_exact("tracking.id_switches", tm1.id_switches, tm2.id_switches)
        check_float("tracking.mota", tm1.mota, tm2.mota)
        check_float("tracking.idf1", tm1.idf1, tm2.idf1)
        check_float("tracking.precision", tm1.precision, tm2.precision)
        check_float("tracking.recall", tm1.recall, tm2.recall)

    # Trajectory statistics
    ts1, ts2 = run1.trajectory_stats, run2.trajectory_stats
    check_exact("trajectory.total_trajectories", ts1.total_trajectories, ts2.total_trajectories)
    check_exact("trajectory.total_observations", ts1.total_observations, ts2.total_observations)
    check_exact("trajectory.trajectories_with_gaps", ts1.trajectories_with_gaps, ts2.trajectories_with_gaps)
    check_exact("trajectory.total_gaps", ts1.total_gaps, ts2.total_gaps)
    check_exact("trajectory.min_gap_frames", ts1.min_gap_frames, ts2.min_gap_frames)
    check_exact("trajectory.max_gap_frames", ts1.max_gap_frames, ts2.max_gap_frames)
    check_float("trajectory.mean_gap_frames", ts1.mean_gap_frames, ts2.mean_gap_frames)
    check_float("trajectory.median_gap_frames", ts1.median_gap_frames, ts2.median_gap_frames)
    check_exact("trajectory.min_lifespan_frames", ts1.min_lifespan_frames, ts2.min_lifespan_frames)
    check_exact("trajectory.max_lifespan_frames", ts1.max_lifespan_frames, ts2.max_lifespan_frames)
    check_float("trajectory.mean_lifespan_frames", ts1.mean_lifespan_frames, ts2.mean_lifespan_frames)
    check_float("trajectory.median_lifespan_frames", ts1.median_lifespan_frames, ts2.median_lifespan_frames)
    check_float("trajectory.frame_coverage_ratio", ts1.frame_coverage_ratio, ts2.frame_coverage_ratio)

    # Image-space analytics
    check_exact("image_spatial.total_detections", run1.image_spatial.total_detections, run2.image_spatial.total_detections)
    check_exact("image_spatial.total_footpoints", run1.image_spatial.total_footpoints, run2.image_spatial.total_footpoints)
    check_exact("image_heatmap.total_accumulated", run1.image_heatmap.total_accumulated, run2.image_heatmap.total_accumulated)
    check_exact("image_heatmap.occupied_cells", run1.image_heatmap.occupied_cells, run2.image_heatmap.occupied_cells)
    check_float("image_heatmap.max_cell_count", run1.image_heatmap.max_cell_count, run2.image_heatmap.max_cell_count)
    check_exact("image_zone.total_memberships", run1.image_zone.total_memberships, run2.image_zone.total_memberships)
    check_exact("image_zone.inside", run1.image_zone.inside_at_least_one_zone, run2.image_zone.inside_at_least_one_zone)
    check_exact("image_zone.outside", run1.image_zone.outside_all_zones, run2.image_zone.outside_all_zones)
    check_exact("image_dwell.total_visits", run1.image_dwell.total_visits, run2.image_dwell.total_visits)
    check_exact("image_dwell.unique_visitors", run1.image_dwell.unique_visitors, run2.image_dwell.unique_visitors)
    check_float("image_dwell.total_dwell_seconds", run1.image_dwell.total_dwell_seconds, run2.image_dwell.total_dwell_seconds)

    # Ground-space analytics
    gp1, gp2 = run1.ground_projection, run2.ground_projection
    check_exact("ground_proj.total_projected", gp1.total_projected, gp2.total_projected)
    check_exact("ground_proj.valid_in_roi", gp1.valid_in_roi, gp2.valid_in_roi)
    check_exact("ground_proj.valid_extrapolated", gp1.valid_extrapolated, gp2.valid_extrapolated)
    check_exact("ground_proj.invalid_projections", gp1.invalid_projections, gp2.invalid_projections)
    check_exact("ground_proj.analytics_accepted", gp1.analytics_accepted, gp2.analytics_accepted)
    check_exact("ground_proj.analytics_excluded", gp1.analytics_excluded, gp2.analytics_excluded)
    check_exact("ground_proj.error_codes", gp1.error_codes, gp2.error_codes)

    check_exact("ground_heatmap.accumulated", run1.ground_heatmap_accumulated, run2.ground_heatmap_accumulated)
    check_exact("ground_heatmap.out_of_bounds", run1.ground_heatmap_out_of_bounds, run2.ground_heatmap_out_of_bounds)
    check_exact("ground_heatmap.invalid", run1.ground_heatmap_invalid, run2.ground_heatmap_invalid)
    check_exact("ground_heatmap.extrap_rejected", run1.ground_heatmap_extrap_rejected, run2.ground_heatmap_extrap_rejected)
    check_exact("ground_heatmap.occupied_cells", run1.ground_heatmap_occupied_cells, run2.ground_heatmap_occupied_cells)
    check_float("ground_heatmap.max_cell_count", run1.ground_heatmap_max_cell_count, run2.ground_heatmap_max_cell_count)

    check_exact("ground_zone.total_memberships", run1.ground_zone_total_memberships, run2.ground_zone_total_memberships)
    check_exact("ground_zone.inside", run1.ground_zone_inside, run2.ground_zone_inside)
    check_exact("ground_zone.outside", run1.ground_zone_outside, run2.ground_zone_outside)
    check_exact("ground_zone.single", run1.ground_zone_single, run2.ground_zone_single)
    check_exact("ground_zone.multi", run1.ground_zone_multi, run2.ground_zone_multi)

    check_exact("ground_dwell.total_visits", run1.ground_dwell_total_visits, run2.ground_dwell_total_visits)
    check_exact("ground_dwell.unique_visitors", run1.ground_dwell_unique_visitors, run2.ground_dwell_unique_visitors)
    check_float("ground_dwell.total_seconds", run1.ground_dwell_total_seconds, run2.ground_dwell_total_seconds)
    check_exact("ground_dwell.visits_by_zone", run1.ground_dwell_visits_by_zone, run2.ground_dwell_visits_by_zone)

    # Cross-space comparison metrics
    cmp1, cmp2 = run1.cross_space_comparison, run2.cross_space_comparison
    check_exact("comparison.image_visits", cmp1.image_visits_total, cmp2.image_visits_total)
    check_exact("comparison.ground_visits", cmp1.ground_visits_total, cmp2.ground_visits_total)
    check_exact("comparison.agreement_count", cmp1.membership_agreement_count, cmp2.membership_agreement_count)
    check_float("comparison.agreement_ratio", cmp1.agreement_ratio, cmp2.agreement_ratio)

    return ReproducibilityReport(
        is_reproducible=len(diffs) == 0,
        sequence_name=run1.sequence_name,
        tolerance_used=abs_tol,
        total_checks=total_checks,
        differences=tuple(diffs),
    )
