"""CLI tool to run multi-object tracking and evaluation on an MOT17 sequence."""

import argparse
import sys
from pathlib import Path
from typing import List

# Add src to sys.path if invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from people_analytics.dataset.mot_sequence import MOT17Sequence
from people_analytics.detection.models import DetectionConfig, PersonDetection
from people_analytics.detection.yolo_detector import YOLOv8PersonDetector
from people_analytics.evaluation.tracking_metrics import evaluate_tracking_sequence
from people_analytics.tracking.byte_tracker import ByteTracker
from people_analytics.tracking.models import TrackerConfig, TrackObservation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-object tracking on an MOT17 sequence.")
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
        help="Detector confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=10,
        help="Maximum frames to process (default: 10, 0 for all)",
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

    print(f"=== Running Tracking on {seq.info.name} (Frames 1 to {end_frame}) ===")
    det_config = DetectionConfig(confidence_threshold=args.conf_threshold)
    detector = YOLOv8PersonDetector(det_config)

    trk_config = TrackerConfig(high_threshold=args.conf_threshold, min_hits=2)
    tracker = ByteTracker(trk_config)

    all_tracks: List[TrackObservation] = []

    for frame_idx in range(1, end_frame + 1):
        frame_path = seq.get_frame_path(frame_idx)
        if not frame_path.is_file():
            continue
        detections = detector.detect_frame(frame_path, frame_index=frame_idx)
        tracks = tracker.update(detections, frame_index=frame_idx)
        all_tracks.extend(tracks)

    unique_track_ids = set(t.track_id for t in all_tracks)
    print(f"Total Track Observations Produced : {len(all_tracks)}")
    print(f"Unique Track IDs Generated       : {len(unique_track_ids)}")

    if seq.has_ground_truth():
        all_gt = seq.load_ground_truth(pedestrians_only=True, active_only=True)
        gt_subset = [g for g in all_gt if 1 <= g.frame <= end_frame]
        metrics = evaluate_tracking_sequence(
            tracks=all_tracks,
            ground_truth=gt_subset,
            iou_threshold=args.iou_threshold,
        )
        print("=== Tracking Evaluation (vs Active Pedestrian GT) ===")
        print(f"  Total Ground Truth : {metrics.total_gt}")
        print(f"  True Positives     : {metrics.true_positives}")
        print(f"  False Positives    : {metrics.false_positives}")
        print(f"  False Negatives    : {metrics.false_negatives}")
        print(f"  ID Switches        : {metrics.id_switches}")
        print(f"  MOTA               : {metrics.mota:.3f}")
        print(f"  IDF1               : {metrics.idf1:.3f}")
        print(f"  Precision          : {metrics.precision:.3f}")
        print(f"  Recall             : {metrics.recall:.3f}")
    print("==================================================")


if __name__ == "__main__":
    main()
