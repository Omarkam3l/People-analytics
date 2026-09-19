"""CLI tool to run person detection and evaluation on an MOT17 sequence."""

import argparse
import sys
from pathlib import Path

# Add src to sys.path if invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from people_analytics.dataset.mot_sequence import MOT17Sequence
from people_analytics.detection.models import DetectionConfig
from people_analytics.detection.yolo_detector import YOLOv8PersonDetector
from people_analytics.evaluation.detection_metrics import evaluate_sequence_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Run person detection on an MOT17 sequence.")
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
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=10,
        help="Maximum frames to process for fast testing (default: 10, 0 for all)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for evaluation matching (default: 0.5)",
    )
    args = parser.parse_args()

    seq = MOT17Sequence(args.sequence_dir)
    end_frame = args.max_frames if args.max_frames > 0 else seq.info.seq_length
    end_frame = min(end_frame, seq.info.seq_length)

    print(f"=== Running Detection on {seq.info.name} (Frames 1 to {end_frame}) ===")
    config = DetectionConfig(confidence_threshold=args.conf_threshold)
    detector = YOLOv8PersonDetector(config)

    detections = detector.detect_sequence(seq, start_frame=1, end_frame=end_frame)
    print(f"Total Detections Generated: {len(detections)}")

    if seq.has_ground_truth():
        all_gt = seq.load_ground_truth(pedestrians_only=True, active_only=True)
        gt_subset = [g for g in all_gt if 1 <= g.frame <= end_frame]
        metrics = evaluate_sequence_detections(
            detections=detections,
            ground_truth=gt_subset,
            iou_threshold=args.iou_threshold,
        )
        print("=== Detection Evaluation (vs Active Pedestrian GT) ===")
        print(f"  True Positives  : {metrics.true_positives}")
        print(f"  False Positives : {metrics.false_positives}")
        print(f"  False Negatives : {metrics.false_negatives}")
        print(f"  Precision       : {metrics.precision:.3f}")
        print(f"  Recall          : {metrics.recall:.3f}")
        print(f"  F1 Score        : {metrics.f1_score:.3f}")
    print("==================================================")


if __name__ == "__main__":
    main()
