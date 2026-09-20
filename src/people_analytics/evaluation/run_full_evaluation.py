"""CLI executable to run full-system evaluation across MOT17-09 and MOT17-02."""

import argparse
import os
from pathlib import Path
import sys
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from people_analytics.dataset.mot_sequence import MOT17Sequence
from people_analytics.detection.models import DetectionConfig
from people_analytics.detection.yolo_detector import YOLOv8PersonDetector
from people_analytics.evaluation.full_pipeline_evaluator import FullPipelineEvaluator
from people_analytics.evaluation.reproducibility import verify_reproducibility
from people_analytics.evaluation.report_generator import (
    generate_conservation_audit_log,
    generate_markdown_report,
    get_report_markdown_path,
    render_visual_artifacts,
    save_machine_readable_results,
)
from people_analytics.evaluation.scene_config import (
    SceneEvaluationConfig,
    get_mot17_02_evaluation_config,
    get_mot17_09_evaluation_config,
)
from people_analytics.footpoint.extractor import BottomCenterFootpointExtractor
from people_analytics.tracking.byte_tracker import ByteTracker
from people_analytics.tracking.models import TrackerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 12: Full-System Dual-Space Pipeline Evaluation")
    parser.add_argument(
        "--sequence",
        "-s",
        type=str,
        default="MOT17-09-FRCNN",
        choices=["MOT17-09-FRCNN", "MOT17-02-FRCNN", "all"],
        help="Sequence to evaluate (default: MOT17-09-FRCNN)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("MOT17/MOT17/train"),
        help="Root path containing MOT17 training sequences",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=0,
        help="Maximum frames to evaluate (default: 0 for complete sequence)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("reports/phase12_evaluation"),
        help="Output directory for reports and artifacts (default: reports/phase12_evaluation)",
    )
    parser.add_argument(
        "--conf-threshold",
        "-c",
        type=float,
        default=0.4,
        help="Detection & tracking confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--reproducibility-check",
        action="store_true",
        help="Execute two consecutive runs to verify AC-08 reproducibility",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export dual-space diagnostic visual artifacts",
    )
    args = parser.parse_args()

    seq_targets = ["MOT17-09-FRCNN", "MOT17-02-FRCNN"] if args.sequence == "all" else [args.sequence]
    reports = []

    det_cfg = DetectionConfig(confidence_threshold=args.conf_threshold)
    trk_cfg = TrackerConfig(high_threshold=args.conf_threshold, min_hits=2)
    footpoint_extractor = BottomCenterFootpointExtractor()

    for seq_name in seq_targets:
        seq_dir = args.data_root / seq_name
        if not seq_dir.is_dir():
            print(f"Error: Sequence directory not found at {seq_dir}", file=sys.stderr)
            sys.exit(1)

        seq = MOT17Sequence(seq_dir)
        config = get_mot17_09_evaluation_config() if "09" in seq_name else get_mot17_02_evaluation_config()

        print(f"\n==================================================================")
        print(f"  PHASE 12 FULL-SYSTEM EVALUATION: {seq_name}")
        print(f"  Total Frames: {seq.info.seq_length} | Evaluating: {args.max_frames or seq.info.seq_length} frames")
        print(f"==================================================================")

        evaluator = FullPipelineEvaluator(
            detector=YOLOv8PersonDetector(det_cfg),
            tracker=ByteTracker(trk_cfg),
            footpoint_extractor=footpoint_extractor,
            config=config,
        )

        report = evaluator.evaluate_sequence(
            sequence=seq,
            max_frames=args.max_frames if args.max_frames > 0 else None,
            evaluate_ground_truth=True,
        )
        reports.append(report)

        # Print quick summary
        print(f"\n[{seq_name} Execution Complete]")
        print(f"  - Wall-clock time: {report.timing.total_ms:.1f} ms ({report.timing.fps:.2f} FPS)")
        print(f"  - Peak RSS memory: {report.peak_rss_mb:.2f} MB")
        if report.detection_metrics:
            print(f"  - Detection F1   : {report.detection_metrics.f1_score:.4f} (TP: {report.detection_metrics.true_positives})")
        if report.tracking_metrics:
            print(f"  - Tracking MOTA  : {report.tracking_metrics.mota:.4f} | IDF1: {report.tracking_metrics.idf1:.4f} (IDSW: {report.tracking_metrics.id_switches})")
        print(f"  - Conservation   : {'ALL PASSED' if report.conservation_audit.all_passed else 'AUDIT FAILURES'}")

        # Save JSON output
        seq_tag = seq_name.lower().replace("-", "_")
        json_path = args.output_dir / f"evaluation_results_{seq_tag}.json"
        save_machine_readable_results(report, json_path)
        print(f"  - Saved JSON     : {json_path}")

        # Save conservation log
        audit_log = generate_conservation_audit_log(report)
        log_path = args.output_dir / f"conservation_audit_{seq_tag}.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(audit_log)
        print(f"  - Saved Audit Log: {log_path}")

        # Optional Reproducibility Verification (AC-08)
        if args.reproducibility_check:
            print(f"\nExecuting second run of {seq_name} for AC-08 reproducibility verification...")
            evaluator2 = FullPipelineEvaluator(
                detector=YOLOv8PersonDetector(det_cfg),
                tracker=ByteTracker(trk_cfg),
                footpoint_extractor=footpoint_extractor,
                config=config,
            )
            report2 = evaluator2.evaluate_sequence(
                sequence=seq,
                max_frames=args.max_frames if args.max_frames > 0 else None,
                evaluate_ground_truth=True,
            )
            repro = verify_reproducibility(report, report2)
            print(f"  - Reproducibility Result: {'REPRODUCIBLE (AC-08 PASSED)' if repro.is_reproducible else 'MISMATCH DETECTED'}")
            if not repro.is_reproducible:
                for d in repro.differences:
                    print(f"    * Difference at {d.field_path}: {d.run1_value} vs {d.run2_value} ({d.difference_type})")

        # Optional Visual Diagnostics
        if args.visualize:
            last_frame_idx = report.frames_processed
            last_img_path = seq.get_frame_path(last_frame_idx)
            if last_img_path.is_file():
                last_bgr = cv2.imread(str(last_img_path))
                frame_state = evaluator.last_frame_state
                comp_p, hm_p = render_visual_artifacts(
                    report=report,
                    last_frame_bgr=last_bgr,
                    tracks=frame_state.tracks if frame_state else [],
                    footpoints=frame_state.footpoints if frame_state else [],
                    ground_obs=frame_state.ground_observations if frame_state else [],
                    image_zones=config.image_zones,
                    ground_zones=config.ground_zones,
                    ground_heatmap_data=frame_state.ground_heatmap_data if frame_state else None,
                    output_dir=args.output_dir,
                    image_memberships=frame_state.image_memberships if frame_state else [],
                )
                print(f"  - Visual Composite: {comp_p}")
                print(f"  - Ground Heatmap  : {hm_p}")

    # Generate Human-Readable Markdown Report
    report_md_path = get_report_markdown_path(args.output_dir, args.sequence)
    generate_markdown_report(reports, report_md_path)
    print(f"\nHuman-Readable Evaluation Report generated at: {report_md_path}")


if __name__ == "__main__":
    main()
