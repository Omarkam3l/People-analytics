"""Command-line tool to inspect and print MOT17 sequence details."""

import argparse
import sys
from pathlib import Path

# Add src to sys.path if invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from people_analytics.dataset.mot_sequence import MOT17Sequence


def inspect_sequence(sequence_dir: Path) -> None:
    """Load and print summary information for a sequence."""
    print(f"=== Inspecting MOT17 Sequence: {sequence_dir} ===")
    seq = MOT17Sequence(sequence_dir)
    info = seq.info

    print(f"Sequence Name : {info.name}")
    print(f"Sequence Length: {info.seq_length} frames")
    print(f"Frame Rate (FPS): {info.frame_rate}")
    print(f"Resolution     : {info.im_width} x {info.im_height}")
    print(f"Image Dir / Ext: {info.im_dir} ({info.im_ext})")
    print(f"Images on Disk : {seq.has_images()}")

    if seq.has_ground_truth():
        all_gt = seq.load_ground_truth()
        ped_gt = seq.load_ground_truth(pedestrians_only=True, active_only=True)
        unique_peds = len(set(g.track_id for g in ped_gt))
        print(f"Ground Truth   : Found ({len(all_gt)} total entries)")
        print(f"  - Active Pedestrian Annotations: {len(ped_gt)}")
        print(f"  - Unique Pedestrian Tracks     : {unique_peds}")
        if ped_gt:
            sample = ped_gt[0]
            print(f"  - Sample Pedestrian Annotation : frame={sample.frame}, track_id={sample.track_id}, "
                  f"bbox={sample.bbox_xywh}, vis={sample.visibility}")
    else:
        print("Ground Truth   : None (Test sequence or missing gt.txt)")

    if seq.has_detections():
        dets = seq.load_detections()
        print(f"Detections     : Found ({len(dets)} total detections)")
        if dets:
            sample_det = dets[0]
            print(f"  - Sample Detection: frame={sample_det.frame}, bbox={sample_det.bbox_xywh}, "
                  f"conf={sample_det.conf:.3f}")
    else:
        print("Detections     : None")

    print("==================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an MOT17 sequence directory.")
    parser.add_argument(
        "--sequence-dir",
        "-s",
        type=Path,
        default=Path("MOT17/MOT17/train/MOT17-09-FRCNN"),
        help="Path to the MOT17 sequence directory (e.g. MOT17/MOT17/train/MOT17-09-FRCNN)",
    )
    args = parser.parse_args()

    if not args.sequence_dir.exists():
        print(f"Error: Directory does not exist: {args.sequence_dir}", file=sys.stderr)
        sys.exit(1)

    inspect_sequence(args.sequence_dir)


if __name__ == "__main__":
    main()
