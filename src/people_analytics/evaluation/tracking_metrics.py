"""Evaluation metrics for multi-object tracking (MOTA, IDF1, ID Switches)."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from people_analytics.dataset.models import GroundTruthAnnotation
from people_analytics.evaluation.detection_metrics import compute_iou
from people_analytics.tracking.models import TrackObservation


@dataclass(frozen=True)
class TrackingMetrics:
    """Multi-object tracking performance metrics."""
    total_gt: int
    true_positives: int
    false_positives: int
    false_negatives: int
    id_switches: int
    mota: float
    idf1: float
    precision: float
    recall: float


def evaluate_tracking_sequence(
    tracks: List[TrackObservation],
    ground_truth: List[GroundTruthAnnotation],
    iou_threshold: float = 0.5,
) -> TrackingMetrics:
    """Evaluate multi-object tracking results against ground truth.

    Evaluation Protocol Note:
        This is a custom tracking evaluation baseline comparing tracker observations against
        active pedestrian ground-truth annotations (class_id == 1, conf == 1.0).
        It is NOT the official MOTChallenge evaluation benchmark (TrackEval), which employs
        specialized suppression and overlap rules for ignore regions and distractors.

    Metrics:
        - True Positives (TP) / False Positives (FP) / False Negatives (FN): Frame-by-frame
          bipartite matching of confirmed track observations to active ground truth with IoU >= iou_threshold.
        - ID Switches (IDSW): Number of times a ground-truth pedestrian's assigned tracker
          hypothesis changes from a previously associated track ID.
        - MOTA (Multiple Object Tracking Accuracy, CLEAR MOT):
          MOTA = 1.0 - (FN + FP + IDSW) / Total_GT
        - IDF1 (Identification F1-Score, Ristani et al. 2016):
          Global optimal bipartite matching maximizing the number of frame matches between predicted
          track identities and ground-truth identities across the full sequence:
          IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    Args:
        tracks: List of TrackObservation objects produced by a tracker.
        ground_truth: List of GroundTruthAnnotation objects for the sequence.
        iou_threshold: Minimum IoU threshold to consider a match (default: 0.5).

    Returns:
        TrackingMetrics summarizing MOTA, IDF1, ID switches, and detection counts.
    """
    # Filter active pedestrians
    active_gt = [g for g in ground_truth if g.is_pedestrian and g.is_active]

    if not active_gt and not tracks:
        return TrackingMetrics(0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0)
    if not active_gt:
        return TrackingMetrics(0, 0, len(tracks), 0, 0, 0.0, 0.0, 0.0, 0.0)
    if not tracks:
        return TrackingMetrics(len(active_gt), 0, 0, len(active_gt), 0, 0.0, 0.0, 0.0, 0.0)

    # Group by frame
    tracks_by_frame: Dict[int, List[TrackObservation]] = defaultdict(list)
    for trk in tracks:
        tracks_by_frame[trk.frame_index].append(trk)

    gt_by_frame: Dict[int, List[GroundTruthAnnotation]] = defaultdict(list)
    for gt in active_gt:
        gt_by_frame[gt.frame].append(gt)

    all_frames = sorted(set(tracks_by_frame.keys()) | set(gt_by_frame.keys()))

    total_gt = len(active_gt)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    id_switches = 0

    # Tracks previous predicted track_id associated with each gt_id
    last_assigned_track: Dict[int, int] = {}

    # Frame-by-frame bipartite matching for MOTA and ID switches
    for frame in all_frames:
        frame_tracks = tracks_by_frame[frame]
        frame_gt = gt_by_frame[frame]

        if not frame_tracks:
            total_fn += len(frame_gt)
            continue
        if not frame_gt:
            total_fp += len(frame_tracks)
            continue

        # IoU matrix: rows = tracks, cols = ground truth
        iou_matrix = np.zeros((len(frame_tracks), len(frame_gt)), dtype=np.float32)
        for t_idx, trk in enumerate(frame_tracks):
            for g_idx, gt in enumerate(frame_gt):
                iou_matrix[t_idx, g_idx] = compute_iou(trk.bbox_xyxy, gt.bbox_xyxy)

        cost_matrix = -iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks: Set[int] = set()
        matched_gts: Set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                matched_tracks.add(r)
                matched_gts.add(c)
                trk_id = frame_tracks[r].track_id
                gt_id = frame_gt[c].track_id

                # Check for ID Switch
                if gt_id in last_assigned_track and last_assigned_track[gt_id] != trk_id:
                    id_switches += 1
                last_assigned_track[gt_id] = trk_id

        tp = len(matched_tracks)
        fp = len(frame_tracks) - tp
        fn = len(frame_gt) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # MOTA calculation
    mota = 1.0 - (total_fn + total_fp + id_switches) / total_gt if total_gt > 0 else 0.0

    # Detection Precision & Recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    # Global trajectory identity matching for IDF1
    idf1 = _compute_idf1(tracks, active_gt, iou_threshold)

    return TrackingMetrics(
        total_gt=total_gt,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        id_switches=id_switches,
        mota=mota,
        idf1=idf1,
        precision=precision,
        recall=recall,
    )


def _compute_idf1(
    tracks: List[TrackObservation],
    active_gt: List[GroundTruthAnnotation],
    iou_threshold: float,
) -> float:
    """Compute IDF1 using global optimal bipartite association between predicted tracks and GT trajectories."""
    # Find all frame-level overlapping pairs
    tracks_by_frame: Dict[int, List[TrackObservation]] = defaultdict(list)
    for trk in tracks:
        tracks_by_frame[trk.frame_index].append(trk)

    gt_by_frame: Dict[int, List[GroundTruthAnnotation]] = defaultdict(list)
    for gt in active_gt:
        gt_by_frame[gt.frame].append(gt)

    unique_track_ids = sorted(set(t.track_id for t in tracks))
    unique_gt_ids = sorted(set(g.track_id for g in active_gt))

    if not unique_track_ids or not unique_gt_ids:
        return 0.0

    track_id_to_idx = {tid: i for i, tid in enumerate(unique_track_ids)}
    gt_id_to_idx = {gid: i for i, gid in enumerate(unique_gt_ids)}

    # Cost / Overlap matrix: overlap_count[track_idx, gt_idx]
    overlap_matrix = np.zeros((len(unique_track_ids), len(unique_gt_ids)), dtype=np.int32)

    for frame, f_tracks in tracks_by_frame.items():
        f_gts = gt_by_frame.get(frame, [])
        if not f_gts:
            continue
        for trk in f_tracks:
            for gt in f_gts:
                if compute_iou(trk.bbox_xyxy, gt.bbox_xyxy) >= iou_threshold:
                    t_idx = track_id_to_idx[trk.track_id]
                    g_idx = gt_id_to_idx[gt.track_id]
                    overlap_matrix[t_idx, g_idx] += 1

    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)

    id_tp = sum(overlap_matrix[r, c] for r, c in zip(row_ind, col_ind))
    id_fp = len(tracks) - id_tp
    id_fn = len(active_gt) - id_tp

    denom = 2 * id_tp + id_fp + id_fn
    return (2.0 * id_tp / denom) if denom > 0 else 0.0
