"""Generators for machine-readable JSON, human-readable Markdown reports, and visual diagnostics."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from people_analytics.evaluation.models import FullSequenceEvaluationReport
from people_analytics.ground_analytics.models import GroundHeatmapData, GroundObservation, GroundZone
from people_analytics.ground_analytics.visualization import (
    draw_dual_view_diagnostics,
    draw_ground_analytics_map,
)
from people_analytics.footpoint.models import FootpointObservation
from people_analytics.pipeline.diagnostics import draw_pipeline_diagnostics
from people_analytics.tracking.models import TrackObservation
from people_analytics.zone.models import Zone, ZoneMembership


def save_machine_readable_results(
    report: FullSequenceEvaluationReport,
    output_path: Path,
) -> None:
    """Serialize evaluation results into a formatted JSON artifact.

    Args:
        report: FullSequenceEvaluationReport instance.
        output_path: Destination JSON file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)


def generate_conservation_audit_log(report: FullSequenceEvaluationReport) -> str:
    """Create a formatted text log verifying all mathematical conservation equations."""
    lines: List[str] = [
        f"================================================================================",
        f"CONSERVATION AUDIT LOG: {report.sequence_name} ({report.frames_processed} frames)",
        f"================================================================================",
        f"{'Invariant ID':<10} | {'Status':<8} | {'Equation Balance':<35} | {'Details'}",
        f"--------------------------------------------------------------------------------",
    ]

    for chk in report.conservation_audit.checks:
        status = "PASSED" if chk.passed else "FAILED"
        balance = f"{chk.left_value} == {chk.right_value}"
        lines.append(f"{chk.name[:10]:<10} | {status:<8} | {balance:<35} | {chk.details}")

    lines.append(f"--------------------------------------------------------------------------------")
    lines.append(f"Overall Audit Status: {'ALL PASSED' if report.conservation_audit.all_passed else 'FAILURES DETECTED'}")
    lines.append(f"================================================================================")
    return "\n".join(lines)


def get_report_markdown_path(output_dir: Path, sequence: str) -> Path:
    """Resolve destination markdown report path for single or consolidated sequence runs.

    Args:
        output_dir: Destination folder for reports.
        sequence: Sequence name (e.g. 'MOT17-09-FRCNN', 'MOT17-02-FRCNN') or 'all'.

    Returns:
        Path to sequence-specific or consolidated markdown report.
    """
    if sequence.lower() == "all":
        return output_dir / "PHASE_12_EVALUATION_REPORT.md"
    seq_short = sequence.replace("-FRCNN", "").replace("-", "_").upper()
    return output_dir / f"PHASE_12_EVALUATION_REPORT_{seq_short}.md"


def generate_markdown_report(
    reports: Sequence[FullSequenceEvaluationReport],
    output_path: Optional[Path] = None,
) -> str:
    """Generate the comprehensive Phase 12 human-readable markdown evaluation report.

    Args:
        reports: Sequence of FullSequenceEvaluationReport objects (primary and secondary).
        output_path: Optional file path to persist the report to disk.

    Returns:
        Formatted markdown string.
    """
    sections: List[str] = [
        "# Phase 12: Full-System Evaluation Report",
        "",
        "## Executive Summary",
        "",
        "This evaluation report validates the complete People Movement Analytics pipeline executed across",
        "full MOT17 benchmark sequences under frozen Phase 1–11 contracts. It provides end-to-end evidence",
        "for object detection, multi-object tracking, footpoint extraction, image-space spatial analytics,",
        "homography projection, ground-plane spatial analytics, and cross-space representation comparisons.",
        "",
        "> [!IMPORTANT]",
        "> **Scientific Scope & Coordinate Limitation**:",
        "> Ground-plane coordinates operate strictly within `CoordinateFrame.ARBITRARY_PLANAR` using reproducible",
        "> controlled manual calibration fixtures. Arbitrary planar units must **NOT** be interpreted as physical meters,",
        "> nor as evidence of higher physical metric accuracy. Cross-space comparison is strictly a coordinate-representation",
        "> evaluation.",
        "",
    ]

    for r in reports:
        sec_title = f"Sequence: `{r.sequence_name}` ({r.frames_processed}/{r.total_sequence_frames} frames @ {r.fps:.1f} FPS, {r.resolution[0]}x{r.resolution[1]})"
        sections.extend([
            f"---",
            f"",
            f"## {sec_title}",
            f"",
            f"### 1. Detection Evaluation (Image Plane)",
            f"- **Protocol**: Phase 2 frozen contract against active pedestrians (`class_id == 1` and `conf == 1.0`).",
            f"- **Matching**: Local greedy one-to-one IoU matching (threshold $\\ge 0.5$). Non-official MOTChallenge benchmark.",
        ])

        if r.detection_metrics:
            dm = r.detection_metrics
            sections.extend([
                f"- **True Positives (TP)**: {dm.true_positives}",
                f"- **False Positives (FP)**: {dm.false_positives}",
                f"- **False Negatives (FN)**: {dm.false_negatives}",
                f"- **Precision**: {dm.precision:.4f}",
                f"- **Recall**: {dm.recall:.4f}",
                f"- **F1 Score**: {dm.f1_score:.4f}",
            ])
        else:
            sections.append(f"- *Ground truth annotations unavailable for detection evaluation.*")

        sections.extend([
            f"",
            f"### 2. Multi-Object Tracking Evaluation",
            f"- **Protocol**: Phase 3 frozen contract against active ground-truth pedestrians.",
            f"- **Matching**: Frame-by-frame Hungarian optimal matching for CLEAR MOT / IDSW; sequence-wide Hungarian for IDF1.",
        ])

        if r.tracking_metrics:
            tm = r.tracking_metrics
            sections.extend([
                f"- **Total GT Pedestrians**: {tm.total_gt}",
                f"- **True Positives (TP)**: {tm.true_positives} | **FP**: {tm.false_positives} | **FN**: {tm.false_negatives}",
                f"- **ID Switches (IDSW)**: {tm.id_switches}",
                f"- **MOTA**: {tm.mota:.4f}",
                f"- **IDF1**: {tm.idf1:.4f}",
                f"- **Tracking Precision**: {tm.precision:.4f} | **Tracking Recall**: {tm.recall:.4f}",
            ])
        else:
            sections.append(f"- *Ground truth annotations unavailable for tracking evaluation.*")

        ts = r.trajectory_stats
        sections.extend([
            f"",
            f"### 3. Trajectory & Motion Dynamics",
            f"- **Total Trajectories**: {ts.total_trajectories}",
            f"- **Total Accumulated Observations**: {ts.total_observations}",
            f"- **Mean Observations per Track**: {ts.mean_observations_per_track:.2f}",
            f"- **Trajectories with Gaps**: {ts.trajectories_with_gaps} (Total Gaps: {ts.total_gaps})",
            f"- **Gap Duration Statistics (frames)**: Min: {ts.min_gap_frames} | Max: {ts.max_gap_frames} | Mean: {ts.mean_gap_frames:.2f} | Median: {ts.median_gap_frames:.1f}",
            f"- **Track Lifespan Statistics (seconds)**: Min: {ts.min_lifespan_seconds:.2f}s | Max: {ts.max_lifespan_seconds:.2f}s | Mean: {ts.mean_lifespan_seconds:.2f}s | Median: {ts.median_lifespan_seconds:.2f}s",
            f"- **Overall Frame Coverage Ratio**: {ts.frame_coverage_ratio:.4f}",
            f"",
            f"### 4. Image-Space Analytics Diagnostics",
            f"- **Heatmap Accumulated**: {r.image_heatmap.total_accumulated} points | **Occupied Cells**: {r.image_heatmap.occupied_cells} ({r.image_heatmap.occupancy_ratio*100:.1f}%) | **Peak Count**: {r.image_heatmap.max_cell_count:.0f}",
            f"- **Zone Query Records**: {r.image_zone.total_memberships} (Inside: {r.image_zone.inside_at_least_one_zone}, Outside: {r.image_zone.outside_all_zones})",
            f"- **Overlapping Zone Breakdown**: Single-zone: {r.image_zone.single_zone_observations} | Multi-zone: {r.image_zone.multi_zone_observations}",
            f"- **Dwell Visits**: {r.image_dwell.total_visits} visits across {r.image_dwell.unique_visitors} unique visitors",
            f"- **Total Dwell Time**: {r.image_dwell.total_dwell_seconds:.2f}s (Average: {r.image_dwell.average_dwell_seconds:.2f}s, Max: {r.image_dwell.max_dwell_seconds:.2f}s)",
            f"",
            f"### 5. Ground-Space Analytics Diagnostics",
            f"- **Coordinate Frame**: `CoordinateFrame.ARBITRARY_PLANAR`",
            f"- **Projection State**: Total: {r.ground_projection.total_projected} | Valid In-ROI: {r.ground_projection.valid_in_roi} | Valid Extrapolated: {r.ground_projection.valid_extrapolated} | Invalid: {r.ground_projection.invalid_projections}",
            f"- **Extrapolation Policy**: `allow_extrapolated=True` (Accepted: {r.ground_projection.analytics_accepted}, Excluded: {r.ground_projection.analytics_excluded})",
            f"- **Error Code Breakdown**: `{r.ground_projection.error_codes}`",
            f"- **Ground Heatmap**: Accumulated: {r.ground_heatmap_accumulated} | Occupied: {r.ground_heatmap_occupied_cells} cells | Peak Count: {r.ground_heatmap_max_cell_count:.0f}",
            f"- **Ground Zone Memberships**: Total: {r.ground_zone_total_memberships} (Inside: {r.ground_zone_inside}, Outside: {r.ground_zone_outside}, Single: {r.ground_zone_single}, Multi: {r.ground_zone_multi})",
            f"- **Ground Dwell Visits**: {r.ground_dwell_total_visits} visits across {r.ground_dwell_unique_visitors} unique visitors (Total: {r.ground_dwell_total_seconds:.2f}s, Average: {r.ground_dwell_average_seconds:.2f}s)",
            f"",
            f"### 6. Cross-Space Representation Comparison",
            f"- **Total Visits**: Image: {r.cross_space_comparison.image_visits_total} vs Ground: {r.cross_space_comparison.ground_visits_total}",
            f"- **Unique Visitors**: Image: {r.cross_space_comparison.image_unique_visitors} vs Ground: {r.cross_space_comparison.ground_unique_visitors}",
            f"- **Total Dwell Duration**: Image: {r.cross_space_comparison.image_dwell_seconds_total:.2f}s vs Ground: {r.cross_space_comparison.ground_dwell_seconds_total:.2f}s",
            f"- **Observation Agreement Ratio**: {r.cross_space_comparison.agreement_ratio*100:.2f}% ({r.cross_space_comparison.membership_agreement_count}/{r.cross_space_comparison.membership_total_evaluated})",
            f"",
            f"### 7. Mathematical Conservation Audit (AC-03 to AC-07)",
            f"| Invariant | Pass/Fail | Equation Balance | Formal Definition |",
            f"| :--- | :---: | :--- | :--- |",
        ])

        for chk in r.conservation_audit.checks:
            st = "✅ PASSED" if chk.passed else "❌ FAILED"
            sections.append(f"| {chk.name} | {st} | `{chk.left_value} == {chk.right_value}` | {chk.details} |")

        t = r.timing
        sections.extend([
            f"",
            f"### 8. Runtime & System Resource Profile (Observational)",
            f"- **Total Wall-Clock Time**: {t.total_ms:.2f} ms ({t.total_ms/1000.0:.2f} s)",
            f"- **Effective Throughput**: **{t.fps:.2f} FPS**",
            f"- **Peak Resident Set Size (RSS)**: {r.peak_rss_mb:.2f} MB",
            f"- **Stage Latency Breakdown**:",
            f"  - Detection: {t.detection_ms:.2f} ms ({t.detection_ms/r.frames_processed:.2f} ms/frame)",
            f"  - Tracking: {t.tracking_ms:.2f} ms ({t.tracking_ms/r.frames_processed:.2f} ms/frame)",
            f"  - Footpoint Extraction: {t.footpoint_ms:.2f} ms ({t.footpoint_ms/r.frames_processed:.2f} ms/frame)",
            f"  - Trajectory Building (Dual): {t.trajectory_ms:.2f} ms ({t.trajectory_ms/r.frames_processed:.2f} ms/frame)",
            f"  - Heatmap Accumulation (Dual): {t.heatmap_ms:.2f} ms ({t.heatmap_ms/r.frames_processed:.2f} ms/frame)",
            f"  - Zone Membership (Dual): {t.zone_ms:.2f} ms ({t.zone_ms/r.frames_processed:.2f} ms/frame)",
            f"  - Dwell Engine (Dual): {t.dwell_ms:.2f} ms ({t.dwell_ms/r.frames_processed:.2f} ms/frame)",
            f"",
        ])

    sections.extend([
        f"---",
        f"",
        f"## Known Limitations & Verification Scope",
        f"1. **Arbitrary Planar Space**: Ground coordinates operate in `ARBITRARY_PLANAR` units and do not denote physical meters.",
        f"2. **Observational Runtime**: Latency, throughput, and memory metrics are descriptive telemetry only and do not establish pass/fail performance gates.",
        f"3. **Local Evaluation Contracts**: Detection and tracking evaluations adhere strictly to frozen custom regression baselines, not official MOTChallenge benchmark submissions.",
        f"",
    ])

    content = "\n".join(sections)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    return content


def render_visual_artifacts(
    report: FullSequenceEvaluationReport,
    last_frame_bgr: np.ndarray,
    tracks: Sequence[TrackObservation],
    footpoints: Sequence[FootpointObservation],
    ground_obs: Sequence[GroundObservation],
    image_zones: Sequence[Zone],
    ground_zones: Sequence[GroundZone],
    ground_heatmap_data: Optional[GroundHeatmapData],
    output_dir: Path,
    image_memberships: Sequence[ZoneMembership] = (),
) -> Tuple[Path, Path]:
    """Generate dual-view composite snapshot and ground heatmap rendering.

    Returns:
        Tuple of (dual_view_composite_path, ground_heatmap_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_tag = report.sequence_name.lower().replace("-", "_")

    # 1. Render Image Perspective View Overlay
    img_overlay = draw_pipeline_diagnostics(
        image=last_frame_bgr,
        tracks=tracks,
        footpoints=footpoints,
        zones=image_zones,
        memberships=image_memberships,
        frame_index=report.frames_processed,
        throughput_fps=report.timing.fps,
    )

    # 2. Render Bird's-Eye Ground Map
    ground_map = draw_ground_analytics_map(
        zones=ground_zones,
        active_observations=ground_obs,
        heatmap_data=ground_heatmap_data,
        canvas_size=(800, 800),
    )

    # 3. Combine into Dual-View Composite
    composite = draw_dual_view_diagnostics(img_overlay, ground_map)
    composite_path = output_dir / f"{seq_tag}_dual_view_composite.png"
    cv2.imwrite(str(composite_path), composite)

    # 4. Render Standalone Ground Heatmap
    heatmap_canvas = draw_ground_analytics_map(
        zones=ground_zones,
        heatmap_data=ground_heatmap_data,
        canvas_size=(800, 800),
    )
    heatmap_path = output_dir / f"{seq_tag}_ground_heatmap.png"
    cv2.imwrite(str(heatmap_path), heatmap_canvas)

    return composite_path, heatmap_path
