"""Standalone ByteTrack implementation for multi-object tracking."""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from people_analytics.detection.models import PersonDetection
from people_analytics.evaluation.detection_metrics import compute_iou
from people_analytics.tracking.base import BaseTracker
from people_analytics.tracking.models import TrackerConfig, TrackObservation, TrackState


@dataclass
class _Track:
    """Internal tracker representation of a tracked target."""
    track_id: int
    bb_left: float
    bb_top: float
    bb_width: float
    bb_height: float
    confidence: float
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    lost_frames: int = 0
    frame_index: int = 1

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        return self.bb_left, self.bb_top, self.bb_left + self.bb_width, self.bb_top + self.bb_height

    def update(self, det: PersonDetection, frame_index: int) -> None:
        self.bb_left = det.bb_left
        self.bb_top = det.bb_top
        self.bb_width = det.bb_width
        self.bb_height = det.bb_height
        self.confidence = det.confidence
        self.hits += 1
        self.lost_frames = 0
        self.frame_index = frame_index


class ByteTracker(BaseTracker):
    """ByteTrack-style multi-object tracker implementing two-stage association.

    Architecture & Fidelity Notes:
        - Classification: ByteTrack-inspired / ByteTrack-style tracker.
        - Association Strategy: Faithfully adopts the defining ByteTrack two-stage data association
          workflow (Zhang et al. ECCV 2022). High-confidence detections are associated first;
          unmatched tracks are then associated with low-confidence detections to recover occluded targets.
        - Motion Model: Direct bounding box IoU. Does NOT incorporate a Kalman filter motion prediction
          model; association cost is computed directly on previous-frame bounding box coordinates.
        - Track Lifecycle Policy: Project-specific TENTATIVE -> CONFIRMED -> LOST -> TERMINATED finite state
          machine designed for downstream spatial analytics:
            - TENTATIVE: New tracks initialized from unmatched high-confidence detections. Promoted to
              CONFIRMED upon reaching min_hits consecutive frame matches.
            - CONFIRMED: Verified tracks. Emitted as TrackObservation when matched in the current frame.
            - LOST: Unmatched confirmed tracks retained in tracker memory for up to max_lost_frames.
              CRITICAL: Zero spatial TrackObservation objects are emitted during LOST state to avoid
              fabricating false coordinates for dwell-time calculations.
            - TERMINATED: Pruned permanently after unobserved frames exceed max_lost_frames.
        - Identity: Strictly monotonic, non-recycling track IDs (1, 2, 3...).

    Reference:
        Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        super().__init__(config)
        self._next_id: int = 1
        self._tracks: List[_Track] = []

    @property
    def tracks(self) -> List[_Track]:
        """Internal list of currently active or lost tracks."""
        return self._tracks

    def reset(self) -> None:
        """Clear all active tracks and reset the ID sequence."""
        self._next_id = 1
        self._tracks.clear()

    def update(
        self,
        detections: List[PersonDetection],
        frame_index: int,
    ) -> List[TrackObservation]:
        """Process detections for the current frame and return confirmed track observations.

        Emission semantics:
            Only CONFIRMED tracks that were successfully matched and updated in the current frame
            are emitted as TrackObservation objects. Lost tracks remain in internal memory for
            potential recovery, but no fabricated spatial observations are generated for frames
            where no valid detection was matched.
        """
        # 1. Separate detections into high-confidence and low-confidence
        high_dets: List[PersonDetection] = []
        low_dets: List[PersonDetection] = []

        for d in detections:
            if d.confidence >= self.config.high_threshold:
                high_dets.append(d)
            elif d.confidence >= self.config.low_threshold:
                low_dets.append(d)

        # 2. Stage 1: Associate all active and lost tracks with high-confidence detections
        stage1_tracks = [
            t for t in self._tracks
            if t.state in (TrackState.CONFIRMED, TrackState.TENTATIVE, TrackState.LOST)
        ]
        matched_s1_track_idx, matched_high_det_idx, unmatched_s1_track_idx, unmatched_high_det_idx = (
            self._associate(stage1_tracks, high_dets, self.config.match_threshold)
        )

        for t_idx, d_idx in zip(matched_s1_track_idx, matched_high_det_idx):
            track = stage1_tracks[t_idx]
            det = high_dets[d_idx]
            track.update(det, frame_index)
            if track.state == TrackState.LOST:
                track.state = TrackState.CONFIRMED
            elif track.state == TrackState.TENTATIVE and track.hits >= self.config.min_hits:
                track.state = TrackState.CONFIRMED

        unmatched_s1_tracks = [stage1_tracks[i] for i in unmatched_s1_track_idx]
        unmatched_high_dets = [high_dets[i] for i in unmatched_high_det_idx]

        # 3. Stage 2: Associate remaining confirmed/lost tracks with low-confidence detections
        stage2_tracks = [t for t in unmatched_s1_tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        matched_s2_track_idx, matched_low_det_idx, unmatched_s2_track_idx, _ = (
            self._associate(stage2_tracks, low_dets, self.config.match_threshold)
        )

        for t_idx, d_idx in zip(matched_s2_track_idx, matched_low_det_idx):
            track = stage2_tracks[t_idx]
            det = low_dets[d_idx]
            track.update(det, frame_index)
            track.state = TrackState.CONFIRMED

        # 4. Handle unmatched tracks
        # Unmatched tentative tracks from stage 1 are terminated
        for track in unmatched_s1_tracks:
            if track.state == TrackState.TENTATIVE:
                track.state = TrackState.TERMINATED

        # Unmatched confirmed/lost tracks from stage 2 become LOST or TERMINATED if exceeded max_lost_frames
        for t_idx in unmatched_s2_track_idx:
            track = stage2_tracks[t_idx]
            track.lost_frames += 1
            if track.lost_frames > self.config.max_lost_frames:
                track.state = TrackState.TERMINATED
            else:
                track.state = TrackState.LOST

        # 5. Initialize new tracks from unmatched high-confidence detections
        for det in unmatched_high_dets:
            initial_state = TrackState.CONFIRMED if self.config.min_hits <= 1 else TrackState.TENTATIVE
            new_track = _Track(
                track_id=self._next_id,
                bb_left=det.bb_left,
                bb_top=det.bb_top,
                bb_width=det.bb_width,
                bb_height=det.bb_height,
                confidence=det.confidence,
                state=initial_state,
                hits=1,
                lost_frames=0,
                frame_index=frame_index,
            )
            self._next_id += 1
            self._tracks.append(new_track)

        # 6. Prune terminated tracks
        self._tracks = [t for t in self._tracks if t.state != TrackState.TERMINATED]

        # 8. Emit confirmed observations for targets updated in the current frame
        observations: List[TrackObservation] = []
        for track in self._tracks:
            if track.state == TrackState.CONFIRMED and track.frame_index == frame_index:
                observations.append(
                    TrackObservation(
                        track_id=track.track_id,
                        frame_index=frame_index,
                        bb_left=track.bb_left,
                        bb_top=track.bb_top,
                        bb_width=track.bb_width,
                        bb_height=track.bb_height,
                        confidence=track.confidence,
                        state=track.state,
                    )
                )

        return observations

    def _associate(
        self,
        tracks: List[_Track],
        detections: List[PersonDetection],
        match_threshold: float,
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Compute optimal linear assignment between tracks and detections based on IoU."""
        if not tracks or not detections:
            return [], [], list(range(len(tracks))), list(range(len(detections)))

        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = compute_iou(track.bbox_xyxy, det.bbox_xyxy)

        # Cost matrix is negative IoU
        cost_matrix = -iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks: List[int] = []
        matched_dets: List[int] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= match_threshold:
                matched_tracks.append(r)
                matched_dets.append(c)
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched_tracks, matched_dets, sorted(unmatched_tracks), sorted(unmatched_dets)
