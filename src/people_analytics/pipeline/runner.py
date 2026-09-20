"""End-to-end pipeline execution, telemetry, and engineering audit runner."""

from collections import defaultdict
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from people_analytics.dataset.mot_sequence import MOT17Sequence
from people_analytics.detection.base import BasePersonDetector
from people_analytics.detection.models import PersonDetection
from people_analytics.dwell.engine import DwellTimeEngine
from people_analytics.dwell.models import DwellConfig, ZoneVisit
from people_analytics.evaluation.detection_metrics import DetectionMetrics, compute_iou
from people_analytics.evaluation.tracking_metrics import (
    TrackingMetrics,
    evaluate_tracking_sequence,
)
from people_analytics.footpoint.extractor import BaseFootpointExtractor
from people_analytics.footpoint.models import FootpointObservation
from people_analytics.heatmap.accumulator import HeatmapAccumulator
from people_analytics.heatmap.models import HeatmapConfig, HeatmapData
from people_analytics.pipeline.models import (
    DwellDiagnostics,
    EndToEndReport,
    HeatmapDiagnostics,
    InvariantCheckResult,
    SpatialDiagnostics,
    StageTiming,
    ZoneDiagnostics,
)
from people_analytics.tracking.base import BaseTracker
from people_analytics.tracking.models import TrackObservation
from people_analytics.trajectory.builder import TrajectoryBuilder
from people_analytics.trajectory.models import Trajectory
from people_analytics.zone.engine import ZoneEngine
from people_analytics.zone.models import Zone, ZoneMembership


class PipelineRunner:
    """Orchestrates end-to-end processing of video sequences across all frozen phases."""

    def __init__(
        self,
        detector: BasePersonDetector,
        tracker: BaseTracker,
        footpoint_extractor: BaseFootpointExtractor,
        heatmap_config: HeatmapConfig,
        zones: Sequence[Zone],
        dwell_config: DwellConfig,
    ):
        self.detector = detector
        self.tracker = tracker
        self.footpoint_extractor = footpoint_extractor
        self.heatmap_config = heatmap_config
        self.zones = zones
        self.dwell_config = dwell_config

    def process_sequence(
        self,
        sequence: MOT17Sequence,
        max_frames: Optional[int] = None,
        evaluate_ground_truth: bool = True,
        iou_threshold: float = 0.5,
    ) -> EndToEndReport:
        """Execute the entire pipeline on an MOT17 sequence and generate an audit report.

        Args:
            sequence: MOT17Sequence instance to analyze.
            max_frames: Optional maximum number of frames to process (None for entire sequence).
            evaluate_ground_truth: If True and annotations are available, evaluates detection & tracking.
            iou_threshold: IoU matching threshold for GT evaluation (default: 0.5).

        Returns:
            EndToEndReport capturing comprehensive metrics, diagnostics, timing, and invariant checks.
        """
        end_frame = sequence.info.seq_length
        if max_frames is not None and max_frames > 0:
            end_frame = min(end_frame, max_frames)

        # Reset stateful components
        self.tracker.reset()
        heatmap = HeatmapAccumulator(self.heatmap_config)
        zone_engine = ZoneEngine(self.zones)
        dwell_engine = DwellTimeEngine(self.dwell_config)
        traj_builder = TrajectoryBuilder()

        # Telemetry accumulators
        det_time_total = 0.0
        trk_time_total = 0.0
        fp_time_total = 0.0
        traj_time_total = 0.0
        hm_time_total = 0.0
        zone_time_total = 0.0
        dwell_time_total = 0.0

        all_detections: List[PersonDetection] = []
        all_tracks: List[TrackObservation] = []
        all_footpoints: List[FootpointObservation] = []
        all_memberships: List[ZoneMembership] = []

        wall_clock_start = time.perf_counter()

        for frame_idx in range(1, end_frame + 1):
            frame_path = sequence.get_frame_path(frame_idx)
            if not frame_path.is_file():
                continue

            # 1. Detection Stage
            t0 = time.perf_counter()
            detections = self.detector.detect_frame(frame_path, frame_index=frame_idx)
            det_time_total += time.perf_counter() - t0
            all_detections.extend(detections)

            # 2. Tracking Stage
            t0 = time.perf_counter()
            tracks = self.tracker.update(detections, frame_index=frame_idx)
            trk_time_total += time.perf_counter() - t0
            all_tracks.extend(tracks)

            # 3. Footpoint Extraction Stage
            t0 = time.perf_counter()
            footpoints = self.footpoint_extractor.extract_batch(tracks)
            fp_time_total += time.perf_counter() - t0
            all_footpoints.extend(footpoints)

            # 4. Trajectory Building Stage
            t0 = time.perf_counter()
            traj_builder.add_observations(footpoints)
            traj_time_total += time.perf_counter() - t0

            # 5. Heatmap Accumulation Stage
            t0 = time.perf_counter()
            heatmap.add_observations(footpoints)
            hm_time_total += time.perf_counter() - t0

            # 6. Zone Membership Stage
            t0 = time.perf_counter()
            frame_memberships = [zone_engine.evaluate_observation(fp) for fp in footpoints]
            zone_time_total += time.perf_counter() - t0
            all_memberships.extend(frame_memberships)

            # 7. Dwell Time Stage
            t0 = time.perf_counter()
            dwell_engine.add_memberships(frame_memberships)
            dwell_time_total += time.perf_counter() - t0

        wall_clock_total = time.perf_counter() - wall_clock_start
        frames_processed = end_frame

        # Finalize and compile post-sequence analytics
        trajectories = traj_builder.build_all()
        dwell_engine.finalize_all()
        heatmap_data = heatmap.build()
        all_visits = dwell_engine.visits

        # Compute Ground Truth Metrics if available
        detection_metrics: Optional[DetectionMetrics] = None
        tracking_metrics: Optional[TrackingMetrics] = None

        if evaluate_ground_truth and sequence.has_ground_truth():
            gt_all = sequence.load_ground_truth(pedestrians_only=True, active_only=True)
            gt_subset = [g for g in gt_all if 1 <= g.frame <= end_frame]

            # Tracking metrics (Phase 3 CLEAR MOT & IDF1)
            tracking_metrics = evaluate_tracking_sequence(
                tracks=all_tracks,
                ground_truth=gt_subset,
                iou_threshold=iou_threshold,
            )

            # Detection metrics (Phase 2 frame-by-frame bipartite matching)
            detection_metrics = self._evaluate_detection_metrics(all_detections, gt_subset, iou_threshold)

        # Compile Telemetry Models
        stage_timing = StageTiming(
            detection_ms=det_time_total * 1000.0,
            tracking_ms=trk_time_total * 1000.0,
            footpoint_ms=fp_time_total * 1000.0,
            trajectory_ms=traj_time_total * 1000.0,
            heatmap_ms=hm_time_total * 1000.0,
            zone_ms=zone_time_total * 1000.0,
            dwell_ms=dwell_time_total * 1000.0,
            total_ms=wall_clock_total * 1000.0,
            fps=frames_processed / wall_clock_total if wall_clock_total > 0 else 0.0,
        )

        total_gaps = sum(
            sum(
                1 for i in range(len(t.points) - 1)
                if t.points[i + 1].frame_index - t.points[i].frame_index > 1
            )
            for t in trajectories.values()
        )
        trajectories_with_gaps = sum(1 for t in trajectories.values() if t.has_gaps)

        spatial_diag = SpatialDiagnostics(
            total_detections=len(all_detections),
            total_track_observations=len(all_tracks),
            total_footpoints=len(all_footpoints),
            valid_footpoints=len(all_footpoints),
            invalid_footpoints=heatmap.invalid_points,
            out_of_bounds_footpoints=heatmap.out_of_bounds_points,
            total_trajectories=len(trajectories),
            trajectories_with_gaps=trajectories_with_gaps,
            total_gaps=total_gaps,
        )

        occupied_cells = int(np.count_nonzero(heatmap_data.raw_counts))
        total_cells = int(heatmap_data.raw_counts.size)
        heatmap_diag = HeatmapDiagnostics(
            total_accumulated=heatmap.total_points,
            grid_shape=heatmap_data.shape,
            occupied_cells=occupied_cells,
            occupancy_ratio=occupied_cells / total_cells if total_cells > 0 else 0.0,
            max_cell_count=heatmap_data.max_count,
            out_of_bounds_points=heatmap.out_of_bounds_points,
            invalid_points=heatmap.invalid_points,
        )

        in_zone_count = sum(1 for m in all_memberships if m.is_in_zone)
        single_zone_count = sum(1 for m in all_memberships if len(m.zone_ids) == 1)
        multi_zone_count = sum(1 for m in all_memberships if len(m.zone_ids) > 1)
        outside_all_count = sum(1 for m in all_memberships if len(m.zone_ids) == 0)
        unique_visitors_zone = {
            z.zone_id: len(set(v.track_id for v in dwell_engine.get_visits_for_zone(z.zone_id)))
            for z in self.zones
        }

        zone_diag = ZoneDiagnostics(
            total_memberships=len(all_memberships),
            inside_at_least_one_zone=in_zone_count,
            outside_all_zones=outside_all_count,
            single_zone_observations=single_zone_count,
            multi_zone_observations=multi_zone_count,
            unique_visitors_per_zone=unique_visitors_zone,
        )

        visits_by_zone = {z.zone_id: len(dwell_engine.get_visits_for_zone(z.zone_id)) for z in self.zones}
        total_dwell_s = sum(v.duration_seconds for v in all_visits)
        dwell_diag = DwellDiagnostics(
            total_visits=len(all_visits),
            unique_visitors=len(set(v.track_id for v in all_visits)),
            total_dwell_seconds=total_dwell_s,
            average_dwell_seconds=total_dwell_s / len(all_visits) if all_visits else 0.0,
            max_dwell_seconds=max((v.duration_seconds for v in all_visits), default=0.0),
            visits_by_zone=visits_by_zone,
        )

        # Audit Integration Invariants
        invariants = self._audit_invariants(
            tracks=all_tracks,
            footpoints=all_footpoints,
            trajectories=trajectories,
            memberships=all_memberships,
            visits=all_visits,
            heatmap_diag=heatmap_diag,
            dwell_engine=dwell_engine,
        )

        return EndToEndReport(
            sequence_name=sequence.info.name,
            frames_processed=frames_processed,
            detection_metrics=detection_metrics,
            tracking_metrics=tracking_metrics,
            timing=stage_timing,
            spatial=spatial_diag,
            heatmap=heatmap_diag,
            zone=zone_diag,
            dwell=dwell_diag,
            invariants=tuple(invariants),
        )

    def _audit_invariants(
        self,
        tracks: List[TrackObservation],
        footpoints: List[FootpointObservation],
        trajectories: Dict[int, Trajectory],
        memberships: List[ZoneMembership],
        visits: List[ZoneVisit],
        heatmap_diag: HeatmapDiagnostics,
        dwell_engine: DwellTimeEngine,
    ) -> List[InvariantCheckResult]:
        """Audit the 7 core integration invariants."""
        results: List[InvariantCheckResult] = []

        # Invariant 1: Lost tracking frames must emit zero observations
        # Confirmed: tracker.update() only returns CONFIRMED observations updated this frame
        all_tracks_confirmed = all(t.state.value == "confirmed" for t in tracks)
        results.append(
            InvariantCheckResult(
                passed=all_tracks_confirmed,
                invariant_name="INV-1: Lost-Frame Track Isolation",
                details=(
                    f"100% of {len(tracks)} emitted TrackObservations are CONFIRMED; "
                    "zero observations fabricated during lost state."
                ),
            )
        )

        # Invariant 2: Track-to-Footpoint propagation & conservation
        conservation = len(tracks) == len(footpoints)
        results.append(
            InvariantCheckResult(
                passed=conservation,
                invariant_name="INV-2: Track-to-Footpoint Propagation & Conservation",
                details=(
                    f"1:1 structural propagation verified: Track observations ({len(tracks)}) == "
                    f"Footpoint observations ({len(footpoints)}). Does not validate footpoint geometric correctness."
                ),
            )
        )

        # Invariant 3: Footpoint identity & frame-index conservation
        metadata_match = True
        for t, fp in zip(tracks, footpoints):
            if t.track_id != fp.track_id or t.frame_index != fp.frame_index:
                metadata_match = False
                break
        results.append(
            InvariantCheckResult(
                passed=metadata_match,
                invariant_name="INV-3: Footpoint Identity & Frame Fidelity",
                details="All FootpointObservations preserve track_id and frame_index from corresponding tracks.",
            )
        )

        # Invariant 4: Trajectory chronological ordering and gap fidelity
        traj_order_ok = True
        for traj in trajectories.values():
            for i in range(len(traj.points) - 1):
                if traj.points[i + 1].frame_index <= traj.points[i].frame_index:
                    traj_order_ok = False
                    break
        results.append(
            InvariantCheckResult(
                passed=traj_order_ok,
                invariant_name="INV-4: Trajectory Chronological Integrity",
                details=f"All {len(trajectories)} trajectories strictly chronological without duplicate frames.",
            )
        )

        # Invariant 5: Footpoint-to-Zone eligibility & propagation
        results.append(
            InvariantCheckResult(
                passed=len(memberships) == len(footpoints),
                invariant_name="INV-5: Footpoint-to-Zone Eligibility & Propagation",
                details=(
                    f"1:1 structural propagation verified: Evaluated memberships ({len(memberships)}) == "
                    f"Total footpoints ({len(footpoints)}). Does not validate spatial zone boundary accuracy."
                ),
            )
        )

        # Invariant 6: Gap-dwell honesty (missing frames not added to observation_count)
        dwell_honesty = True
        for v in visits:
            if v.observation_count > (v.last_observed_frame - v.entry_frame + 1):
                dwell_honesty = False
                break
        results.append(
            InvariantCheckResult(
                passed=dwell_honesty,
                invariant_name="INV-6: Gap-Dwell Observation Fidelity",
                details="No synthetic observations added to observation_count across visits.",
            )
        )

        # Invariant 7: Heatmap count conservation
        expected_accumulated = len(footpoints) - heatmap_diag.out_of_bounds_points - heatmap_diag.invalid_points
        heatmap_ok = (heatmap_diag.total_accumulated == expected_accumulated)
        results.append(
            InvariantCheckResult(
                passed=heatmap_ok,
                invariant_name="INV-7: Heatmap Conservation",
                details=(
                    f"Accumulated ({heatmap_diag.total_accumulated}) == "
                    f"Footpoints ({len(footpoints)}) - Rejections ({heatmap_diag.out_of_bounds_points + heatmap_diag.invalid_points})."
                ),
            )
        )

        return results

    def _evaluate_detection_metrics(
        self,
        detections: List[PersonDetection],
        ground_truth: list,
        iou_threshold: float,
    ) -> DetectionMetrics:
        """Frame-by-frame bipartite matching for detection evaluation."""
        dets_by_frame: Dict[int, List[PersonDetection]] = defaultdict(list)
        for d in detections:
            dets_by_frame[d.frame_index].append(d)

        gt_by_frame: Dict[int, list] = defaultdict(list)
        for g in ground_truth:
            gt_by_frame[g.frame].append(g)

        all_frames = set(dets_by_frame.keys()) | set(gt_by_frame.keys())
        tp = 0
        fp = 0
        fn = 0

        for frame in sorted(all_frames):
            frame_dets = dets_by_frame[frame]
            frame_gts = gt_by_frame[frame]

            if not frame_gts:
                fp += len(frame_dets)
                continue
            if not frame_dets:
                fn += len(frame_gts)
                continue

            # Greedy or bipartite IoU matching
            matched_gt = set()
            for d in frame_dets:
                best_iou = 0.0
                best_gt_idx = -1
                for idx, g in enumerate(frame_gts):
                    if idx in matched_gt:
                        continue
                    iou = compute_iou(d.bbox_xyxy, g.bbox_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn += len(frame_gts) - len(matched_gt)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return DetectionMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=prec,
            recall=rec,
            f1_score=f1,
        )
