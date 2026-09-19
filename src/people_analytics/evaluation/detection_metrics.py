"""Detection evaluation metrics and IoU matching against ground truth."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from people_analytics.dataset.models import GroundTruthAnnotation
from people_analytics.detection.models import PersonDetection


@dataclass(frozen=True)
class DetectionMetrics:
    """Metrics assessing person detector performance against ground truth."""
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float

    @property
    def total_predictions(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def total_ground_truth(self) -> int:
        return self.true_positives + self.false_negatives


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union (IoU) between two boxes in (x1, y1, x2, y2) format.

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        IoU score in range [0.0, 1.0].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0.0, x2 - x1)
    inter_height = max(0.0, y2 - y1)
    intersection = inter_width * inter_height

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0

    return intersection / union


def evaluate_detections_frame(
    detections: List[PersonDetection],
    ground_truth: List[GroundTruthAnnotation],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Evaluate detections against ground truth for a single frame using greedy IoU-based one-to-one matching.

    Evaluation protocol note:
        This is a custom pedestrian detection evaluation baseline comparing live detector outputs
        against active pedestrian ground-truth annotations (class_id == 1, conf == 1.0).
        It is NOT the official MOTChallenge evaluation protocol, which employs specialized
        suppression/overlap rules for ignore regions, occluders, and distractors.

    Matching algorithm:
        Greedy IoU-based one-to-one matching: Candidate pairs with IoU >= iou_threshold are sorted
        descending by IoU score and greedily associated such that each detection and ground-truth
        annotation is matched at most once.

    Args:
        detections: List of PersonDetection instances for this frame.
        ground_truth: List of GroundTruthAnnotation instances for this frame.
        iou_threshold: Minimum IoU to consider a match a True Positive.

    Returns:
        DetectionMetrics for this frame.
    """
    active_gt = [g for g in ground_truth if g.is_pedestrian and g.is_active]

    if not detections and not active_gt:
        return DetectionMetrics(0, 0, 0, 1.0, 1.0, 1.0)

    if not detections:
        return DetectionMetrics(0, 0, len(active_gt), 0.0, 0.0, 0.0)

    if not active_gt:
        return DetectionMetrics(0, len(detections), 0, 0.0, 0.0, 0.0)

    # Compute candidate IoUs
    candidates: List[Tuple[float, int, int]] = []
    for d_idx, det in enumerate(detections):
        for g_idx, gt in enumerate(active_gt):
            iou = compute_iou(det.bbox_xyxy, gt.bbox_xyxy)
            if iou >= iou_threshold:
                candidates.append((iou, d_idx, g_idx))

    # Sort candidates by descending IoU for greedy matching
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_dets: Set[int] = set()
    matched_gts: Set[int] = set()

    for _, d_idx, g_idx in candidates:
        if d_idx not in matched_dets and g_idx not in matched_gts:
            matched_dets.add(d_idx)
            matched_gts.add(g_idx)

    tp = len(matched_dets)
    fp = len(detections) - tp
    fn = len(active_gt) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return DetectionMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )


def evaluate_sequence_detections(
    detections: List[PersonDetection],
    ground_truth: List[GroundTruthAnnotation],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Evaluate detections aggregated across all frames of a sequence.

    Args:
        detections: List of PersonDetection instances across the sequence.
        ground_truth: List of GroundTruthAnnotation instances across the sequence.
        iou_threshold: Minimum IoU threshold for a match.

    Returns:
        Aggregated DetectionMetrics for the sequence.
    """
    dets_by_frame: Dict[int, List[PersonDetection]] = defaultdict(list)
    for det in detections:
        dets_by_frame[det.frame_index].append(det)

    gt_by_frame: Dict[int, List[GroundTruthAnnotation]] = defaultdict(list)
    for gt in ground_truth:
        if gt.is_pedestrian and gt.is_active:
            gt_by_frame[gt.frame].append(gt)

    all_frames = set(dets_by_frame.keys()) | set(gt_by_frame.keys())

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for frame in sorted(all_frames):
        frame_metrics = evaluate_detections_frame(
            detections=dets_by_frame[frame],
            ground_truth=gt_by_frame[frame],
            iou_threshold=iou_threshold,
        )
        total_tp += frame_metrics.true_positives
        total_fp += frame_metrics.false_positives
        total_fn += frame_metrics.false_negatives

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return DetectionMetrics(
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )
