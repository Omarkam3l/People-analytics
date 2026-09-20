"""CLI entry point to execute end-to-end evaluation and audit on an MOT17 sequence."""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from people_analytics.dataset.mot_sequence import MOT17Sequence
from people_analytics.detection.models import DetectionConfig
from people_analytics.detection.yolo_detector import YOLOv8PersonDetector
from people_analytics.dwell.models import DwellConfig
from people_analytics.footpoint.extractor import BottomCenterFootpointExtractor
from people_analytics.heatmap.models import HeatmapConfig
from people_analytics.pipeline.runner import PipelineRunner
from people_analytics.tracking.byte_tracker import ByteTracker
from people_analytics.tracking.models import TrackerConfig
from people_analytics.zone.models import Zone


def create_default_zones(image_width: int, image_height: int) -> list[Zone]:
    """Generate representative spatial zones scaled to image canvas dimensions."""
    # Zone 1: Sidewalk / Central Corridor
    z1 = Zone(
        zone_id="Zone_Central",
        name="Central Corridor",
        vertices=(
            (0.2 * image_width, 0.4 * image_height),
            (0.8 * image_width, 0.4 * image_height),
            (0.8 * image_width, 0.8 * image_height),
            (0.2 * image_width, 0.8 * image_height),
        ),
    )
    # Zone 2: Entrance / Left Periphery
    z2 = Zone(
        zone_id="Zone_Left",
        name="Left Periphery",
        vertices=(
            (0.0 * image_width, 0.3 * image_height),
            (0.35 * image_width, 0.3 * image_height),
            (0.35 * image_width, 0.9 * image_height),
            (0.0 * image_width, 0.9 * image_height),
        ),
    )
    return [z1, z2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run End-to-End People Movement Analytics Pipeline.")
    parser.add_argument(
        "--sequence-dir",
        "-s",
        type=Path,
        default=Path("MOT17/MOT17/train/MOT17-09-FRCNN"),
        help="Path to MOT17 sequence directory",
    )
    parser.add_argument(
        "--conf-threshold",
        "-c",
        type=float,
        default=0.4,
        help="Detection & tracking confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=50,
        help="Maximum frames to process (default: 50, 0 for all)",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=50,
        help="Heatmap discrete grid cell size in pixels (default: 50)",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=0,
        help="Dwell engine gap tolerance in frames (default: 0 for strict observation)",
    )
    args = parser.parse_args()

    seq = MOT17Sequence(args.sequence_dir)
    width = seq.info.im_width
    height = seq.info.im_height
    fps = seq.info.frame_rate

    is_full_sequence = (args.max_frames == 0 or args.max_frames >= seq.info.seq_length)
    if is_full_sequence:
        scope_title = "FULL-SEQUENCE EVALUATION (SCOPE C)"
    elif "MOT17-09" in seq.info.name:
        scope_title = f"PRIMARY INTEGRATION EVALUATION (SCOPE B: {min(args.max_frames, seq.info.seq_length)} frames)"
    else:
        scope_title = f"SECONDARY SMOKE VALIDATION (SCOPE B: {min(args.max_frames, seq.info.seq_length)} frames — Smoke/Sanity Only, NOT Generalization Evidence)"

    print(f"==================================================================")
    print(f"  PEOPLE MOVEMENT ANALYTICS — PIPELINE AUDIT & EVALUATION")
    print(f"  Evaluation Scope : {scope_title}")
    print(f"  Sequence         : {seq.info.name} ({width}x{height} @ {fps} FPS, Total Sequence Length: {seq.info.seq_length} frames)")
    print(f"==================================================================")

    # Initialize components
    det_config = DetectionConfig(confidence_threshold=args.conf_threshold)
    detector = YOLOv8PersonDetector(det_config)
    trk_config = TrackerConfig(high_threshold=args.conf_threshold, min_hits=2)
    tracker = ByteTracker(trk_config)
    footpoint_extractor = BottomCenterFootpointExtractor()
    heatmap_config = HeatmapConfig(image_width=width, image_height=height, cell_size=args.cell_size)
    zones = create_default_zones(width, height)
    dwell_config = DwellConfig(fps=fps, max_gap_frames=args.max_gap_frames)

    runner = PipelineRunner(
        detector=detector,
        tracker=tracker,
        footpoint_extractor=footpoint_extractor,
        heatmap_config=heatmap_config,
        zones=zones,
        dwell_config=dwell_config,
    )

    report = runner.process_sequence(
        sequence=seq,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        evaluate_ground_truth=True,
    )

    eval_duration_s = report.frames_processed / fps if fps > 0 else 0.0

    # Print Audit Report
    print(f"\n[1] Performance & Latency Breakdown ({report.frames_processed} frames evaluated):")
    print(f"    - Detection     : {report.timing.detection_ms:8.2f} ms ({report.timing.detection_ms/report.frames_processed:6.2f} ms/frame)")
    print(f"    - Tracking      : {report.timing.tracking_ms:8.2f} ms ({report.timing.tracking_ms/report.frames_processed:6.2f} ms/frame)")
    print(f"    - Footpoint     : {report.timing.footpoint_ms:8.2f} ms ({report.timing.footpoint_ms/report.frames_processed:6.2f} ms/frame)")
    print(f"    - Trajectory    : {report.timing.trajectory_ms:8.2f} ms ({report.timing.trajectory_ms/report.frames_processed:6.2f} ms/frame)")
    print(f"    - Heatmap       : {report.timing.heatmap_ms:8.2f} ms ({report.timing.heatmap_ms/report.frames_processed:6.2f} ms/frame)")
    print(f"    - Zone Eval     : {report.timing.zone_ms:8.2f} ms ({report.timing.zone_ms/report.frames_processed:6.2f} ms/frame)")
    print(f"    - Dwell Time    : {report.timing.dwell_ms:8.2f} ms ({report.timing.dwell_ms/report.frames_processed:6.2f} ms/frame)")
    print(f"    - Total Time    : {report.timing.total_ms:8.2f} ms")
    print(f"    - Throughput    : {report.timing.fps:8.2f} FPS")

    if report.detection_metrics:
        print(f"\n[2] Detection Evaluation (IoU >= 0.5 vs Ground Truth on Evaluated Window):")
        print(f"    - TP: {report.detection_metrics.true_positives} | FP: {report.detection_metrics.false_positives} | FN: {report.detection_metrics.false_negatives}")
        print(f"    - Precision: {report.detection_metrics.precision:.3f} | Recall: {report.detection_metrics.recall:.3f} | F1-Score: {report.detection_metrics.f1_score:.3f}")

    if report.tracking_metrics:
        print(f"\n[3] Tracking Evaluation (CLEAR MOT & IDF1 vs Active GT on Evaluated Window):")
        print(f"    - MOTA: {report.tracking_metrics.mota:.3f} | IDF1: {report.tracking_metrics.idf1:.3f} | ID Switches: {report.tracking_metrics.id_switches}")
        print(f"    - Precision: {report.tracking_metrics.precision:.3f} | Recall: {report.tracking_metrics.recall:.3f}")

    print(f"\n[4] Spatial Conservation Diagnostics:")
    print(f"    - Total Detections             : {report.spatial.total_detections}")
    print(f"    - Confirmed Track Observations : {report.spatial.total_track_observations}")
    print(f"    - Extracted Footpoints         : {report.spatial.total_footpoints}")
    print(f"    - Trajectories                 : {report.spatial.total_trajectories} (With gaps: {report.spatial.trajectories_with_gaps}, Total gaps: {report.spatial.total_gaps})")
    print(f"    --- Conservation Equation 1: Footpoints vs Heatmap ---")
    print(f"        Footpoints ({report.spatial.total_footpoints}) = "
          f"Accumulated ({report.heatmap.total_accumulated}) + "
          f"Out-of-Bounds ({report.heatmap.out_of_bounds_points}) + "
          f"Invalid ({report.heatmap.invalid_points}) "
          f"[{'BALANCED' if report.spatial.total_footpoints == report.heatmap.total_accumulated + report.heatmap.out_of_bounds_points + report.heatmap.invalid_points else 'UNBALANCED'}]")
    print(f"    --- Conservation Equation 2: Footpoints vs Spatial Zones ---")
    print(f"        Footpoints ({report.spatial.total_footpoints}) = "
          f"Inside At Least One Zone ({report.zone.inside_at_least_one_zone}) + "
          f"Outside All Zones ({report.zone.outside_all_zones}) "
          f"[{'BALANCED' if report.spatial.total_footpoints == report.zone.inside_at_least_one_zone + report.zone.outside_all_zones else 'UNBALANCED'}]")

    print(f"\n[5] Spatial Zones Membership Breakdown:")
    print(f"    - Total ZoneMembership Records : {report.zone.total_memberships} (1:1 with Footpoint observations)")
    print(f"    - Single-Zone Memberships      : {report.zone.single_zone_observations}")
    print(f"    - Multi-Zone Memberships       : {report.zone.multi_zone_observations}")
    print(f"    - Outside-All-Zones Records    : {report.zone.outside_all_zones}")
    print(f"    - Inside At Least One Zone     : {report.zone.inside_at_least_one_zone}")
    print(f"      (Note: 'ZoneMembership records' denote per-footpoint query structures; 'points inside zones' = single + multi)")

    print(f"\n[6] Heatmap Distribution Diagnostics:")
    print(f"    - Grid Shape    : {report.heatmap.grid_shape} (Cell size: {args.cell_size}px)")
    print(f"    - Accumulated   : {report.heatmap.total_accumulated} points")
    print(f"    - Occupancy     : {report.heatmap.occupied_cells} cells ({report.heatmap.occupancy_ratio*100:.1f}%) | Peak count: {report.heatmap.max_cell_count:.0f}")

    print(f"\n[7] Dwell Analytics (Evaluated Window Only: frames 1 to {report.frames_processed}, {eval_duration_s:.2f}s):")
    print(f"    [IMPORTANT: The following metrics reflect ONLY the {report.frames_processed}-frame evaluated window, NOT full-sequence scene totals]")
    print(f"    - Total Visits in Window       : {report.dwell.total_visits}")
    print(f"    - Unique Visitors in Window    : {report.dwell.unique_visitors}")
    print(f"    - Total Dwell Time in Window   : {report.dwell.total_dwell_seconds:.2f}s")
    print(f"    - Average Dwell Time per Visit : {report.dwell.average_dwell_seconds:.2f}s")
    print(f"    - Max Dwell Time in Window     : {report.dwell.max_dwell_seconds:.2f}s")
    print(f"    - Visits by Zone in Window     : {report.dwell.visits_by_zone}")

    print(f"\n[8] Integration Invariants Audit:")
    for inv in report.invariants:
        status = "PASSED" if inv.passed else "FAILED"
        print(f"    [{status}] {inv.invariant_name}: {inv.details}")

    print(f"\nOverall Audit Status: {'ALL INVARIANTS PASSED' if report.all_invariants_passed else 'AUDIT FAILURES DETECTED'}")
    print(f"==================================================================")
    print(f"==================================================================")


if __name__ == "__main__":
    main()
