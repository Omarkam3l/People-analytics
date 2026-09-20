"""Full-pipeline evaluator executing the complete dual-space analytics pipeline."""

from collections import defaultdict
import os
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

from people_analytics.dataset.mot_sequence import MOT17Sequence
from people_analytics.detection.base import BasePersonDetector
from people_analytics.detection.models import PersonDetection
from people_analytics.dwell.engine import DwellTimeEngine
from people_analytics.dwell.models import ZoneVisit
from people_analytics.evaluation.detection_metrics import DetectionMetrics, compute_iou
from people_analytics.evaluation.models import (
    ConservationCheck,
    EvaluationFrameState,
    FullConservationAudit,
    FullSequenceEvaluationReport,
    GroundProjectionDiagnostics,
    TrajectorySummaryStats,
)
from people_analytics.evaluation.scene_config import SceneEvaluationConfig
from people_analytics.evaluation.tracking_metrics import (
    TrackingMetrics,
    evaluate_tracking_sequence,
)
from people_analytics.footpoint.extractor import BaseFootpointExtractor
from people_analytics.footpoint.models import FootpointObservation
from people_analytics.ground.projector import GroundPlaneProjector
from people_analytics.ground_analytics import (
    GroundDwellEngine,
    GroundHeatmapAccumulator,
    GroundHeatmapData,
    GroundObservation,
    GroundTrajectory,
    GroundTrajectoryBuilder,
    GroundZoneEngine,
    GroundZoneMembership,
    compare_image_and_ground_analytics,
)
from people_analytics.heatmap.accumulator import HeatmapAccumulator
from people_analytics.pipeline.models import (
    DwellDiagnostics,
    HeatmapDiagnostics,
    SpatialDiagnostics,
    StageTiming,
    ZoneDiagnostics,
)
from people_analytics.tracking.base import BaseTracker
from people_analytics.tracking.models import TrackObservation
from people_analytics.trajectory.builder import TrajectoryBuilder
from people_analytics.trajectory.models import Trajectory
from people_analytics.zone.engine import ZoneEngine
from people_analytics.zone.models import ZoneMembership


class FullPipelineEvaluator:
    """Orchestrates end-to-end execution of the complete dual-space analytics pipeline."""

    def __init__(
        self,
        detector: BasePersonDetector,
        tracker: BaseTracker,
        footpoint_extractor: BaseFootpointExtractor,
        config: SceneEvaluationConfig,
    ):
        self.detector = detector
        self.tracker = tracker
        self.footpoint_extractor = footpoint_extractor
        self.config = config
        self.last_frame_state: Optional[EvaluationFrameState] = None

    @property
    def last_frame_tracks(self) -> List[TrackObservation]:
        """Active track observations from the final evaluated frame."""
        return self.last_frame_state.tracks if self.last_frame_state else []

    @property
    def last_frame_footpoints(self) -> List[FootpointObservation]:
        """Extracted footpoint observations from the final evaluated frame."""
        return self.last_frame_state.footpoints if self.last_frame_state else []

    @property
    def last_frame_img_memberships(self) -> List[ZoneMembership]:
        """Image zone memberships from the final evaluated frame."""
        return self.last_frame_state.image_memberships if self.last_frame_state else []

    @property
    def last_frame_ground_obs(self) -> List[GroundObservation]:
        """Ground observations from the final evaluated frame."""
        return self.last_frame_state.ground_observations if self.last_frame_state else []

    @property
    def last_ground_heatmap_data(self) -> Optional[GroundHeatmapData]:
        """Ground heatmap accumulated density data."""
        return self.last_frame_state.ground_heatmap_data if self.last_frame_state else None

    def evaluate_sequence(
        self,
        sequence: MOT17Sequence,
        max_frames: Optional[int] = None,
        evaluate_ground_truth: bool = True,
        iou_threshold: float = 0.5,
    ) -> FullSequenceEvaluationReport:
        """Run full evaluation on an MOT17 sequence and produce a FullSequenceEvaluationReport.

        Args:
            sequence: Loaded MOT17Sequence to evaluate.
            max_frames: Optional frame count limit (None for all frames).
            evaluate_ground_truth: If True, evaluates detection & tracking against ground truth.
            iou_threshold: IoU matching threshold (default: 0.5).

        Returns:
            FullSequenceEvaluationReport capturing metrics across both image and ground spaces.
        """
        end_frame = sequence.info.seq_length
        if max_frames is not None and max_frames > 0:
            end_frame = min(end_frame, max_frames)

        # Reset stateful tracker
        self.tracker.reset()

        # Initialize Image-Space Engines
        img_heatmap = HeatmapAccumulator(self.config.image_heatmap_config)
        img_zone_engine = ZoneEngine(self.config.image_zones)
        img_dwell_engine = DwellTimeEngine(self.config.dwell_config)
        img_traj_builder = TrajectoryBuilder()

        # Initialize Ground-Space Engines (preserves Phase 11 default: allow_extrapolated=True)
        projector = GroundPlaneProjector(self.config.ground_calibration)
        grd_traj_builder = GroundTrajectoryBuilder(allow_extrapolated=True)
        grd_heatmap = GroundHeatmapAccumulator(self.config.ground_heatmap_config)
        grd_zone_engine = GroundZoneEngine(self.config.ground_zones, allow_extrapolated=True)
        grd_dwell_engine = GroundDwellEngine(self.config.dwell_config)

        # Telemetry accumulators (seconds)
        det_time = 0.0
        trk_time = 0.0
        fp_time = 0.0
        img_traj_time = 0.0
        img_hm_time = 0.0
        img_zone_time = 0.0
        img_dwell_time = 0.0
        proj_time = 0.0
        grd_traj_time = 0.0
        grd_hm_time = 0.0
        grd_zone_time = 0.0
        grd_dwell_time = 0.0

        all_detections: List[PersonDetection] = []
        all_tracks: List[TrackObservation] = []
        all_footpoints: List[FootpointObservation] = []
        all_img_memberships: List[ZoneMembership] = []
        all_grd_observations: List[GroundObservation] = []
        all_grd_memberships: List[GroundZoneMembership] = []

        last_frame_evaluated: int = 0
        last_tracks: List[TrackObservation] = []
        last_footpoints: List[FootpointObservation] = []
        last_img_memberships: List[ZoneMembership] = []
        last_grd_obs: List[GroundObservation] = []
        last_grd_memberships: List[GroundZoneMembership] = []

        wall_start = time.perf_counter()

        for frame_idx in range(1, end_frame + 1):
            frame_path = sequence.get_frame_path(frame_idx)
            if not frame_path.is_file():
                continue

            last_frame_evaluated = frame_idx

            # 1. Detection
            t0 = time.perf_counter()
            detections = self.detector.detect_frame(frame_path, frame_index=frame_idx)
            det_time += time.perf_counter() - t0
            all_detections.extend(detections)

            # 2. Tracking
            t0 = time.perf_counter()
            tracks = self.tracker.update(detections, frame_index=frame_idx)
            trk_time += time.perf_counter() - t0
            all_tracks.extend(tracks)
            last_tracks = tracks

            # 3. Footpoint Extraction
            t0 = time.perf_counter()
            footpoints = self.footpoint_extractor.extract_batch(tracks)
            fp_time += time.perf_counter() - t0
            all_footpoints.extend(footpoints)
            last_footpoints = footpoints

            # 4. Image-Space Trajectory
            t0 = time.perf_counter()
            img_traj_builder.add_observations(footpoints)
            img_traj_time += time.perf_counter() - t0

            # 5. Image-Space Heatmap
            t0 = time.perf_counter()
            img_heatmap.add_observations(footpoints)
            img_hm_time += time.perf_counter() - t0

            # 6. Image-Space Zone
            t0 = time.perf_counter()
            frame_img_memberships = [img_zone_engine.evaluate_observation(fp) for fp in footpoints]
            img_zone_time += time.perf_counter() - t0
            all_img_memberships.extend(frame_img_memberships)
            last_img_memberships = frame_img_memberships

            # 7. Image-Space Dwell
            t0 = time.perf_counter()
            img_dwell_engine.add_memberships(frame_img_memberships)
            img_dwell_time += time.perf_counter() - t0

            # 8. Ground-Plane Homography Projection
            t0 = time.perf_counter()
            frame_grd_obs = [projector.project_observation(fp) for fp in footpoints]
            proj_time += time.perf_counter() - t0
            all_grd_observations.extend(frame_grd_obs)
            last_grd_obs = frame_grd_obs

            # 9. Ground-Space Trajectory
            t0 = time.perf_counter()
            grd_traj_builder.add_observations(frame_grd_obs)
            grd_traj_time += time.perf_counter() - t0

            # 10. Ground-Space Heatmap
            t0 = time.perf_counter()
            grd_heatmap.add_observations(frame_grd_obs)
            grd_hm_time += time.perf_counter() - t0

            # 11. Ground-Space Zones
            t0 = time.perf_counter()
            frame_grd_memberships = [grd_zone_engine.evaluate_observation(o) for o in frame_grd_obs]
            grd_zone_time += time.perf_counter() - t0
            all_grd_memberships.extend(frame_grd_memberships)
            last_grd_memberships = frame_grd_memberships

            # 12. Ground-Space Dwell
            t0 = time.perf_counter()
            grd_dwell_engine.add_memberships(frame_grd_memberships)
            grd_dwell_time += time.perf_counter() - t0

        wall_total = time.perf_counter() - wall_start
        frames_processed = end_frame

        # Finalize and compile post-sequence analytics
        img_trajectories = img_traj_builder.build_all()
        img_dwell_engine.finalize_all()
        img_heatmap_data = img_heatmap.build()
        img_visits = img_dwell_engine.visits

        grd_trajectories = grd_traj_builder.build_all()
        grd_dwell_engine.finalize_all()
        grd_heatmap_data = grd_heatmap.build()
        grd_visits = grd_dwell_engine.visits

        # Save active final frame state for visualization diagnostics
        self.last_frame_state = EvaluationFrameState(
            frame_index=last_frame_evaluated,
            tracks=last_tracks,
            footpoints=last_footpoints,
            image_memberships=last_img_memberships,
            ground_observations=last_grd_obs,
            ground_memberships=last_grd_memberships,
            ground_heatmap_data=grd_heatmap_data,
        )

        # Ground Truth Metrics (using frozen Phase 2 and Phase 3 evaluation contracts)
        detection_metrics: Optional[DetectionMetrics] = None
        tracking_metrics: Optional[TrackingMetrics] = None

        if evaluate_ground_truth and sequence.has_ground_truth():
            gt_all = sequence.load_ground_truth(pedestrians_only=True, active_only=True)
            gt_subset = [g for g in gt_all if 1 <= g.frame <= end_frame]

            tracking_metrics = evaluate_tracking_sequence(
                tracks=all_tracks,
                ground_truth=gt_subset,
                iou_threshold=iou_threshold,
            )
            detection_metrics = self._evaluate_detection_metrics(all_detections, gt_subset, iou_threshold)

        # Cross-space representation comparison
        comparison_report = compare_image_and_ground_analytics(
            image_memberships=all_img_memberships,
            ground_memberships=all_grd_memberships,
            image_visits=img_visits,
            ground_visits=grd_visits,
        )

        # Trajectory statistics computation
        traj_stats = self._compute_trajectory_stats(img_trajectories, self.config.fps)

        # Peak RSS Memory measurement (observational only)
        peak_rss_mb = self._get_peak_rss_mb()

        # Telemetry timing models
        stage_timing = StageTiming(
            detection_ms=det_time * 1000.0,
            tracking_ms=trk_time * 1000.0,
            footpoint_ms=fp_time * 1000.0,
            trajectory_ms=(img_traj_time + grd_traj_time) * 1000.0,
            heatmap_ms=(img_hm_time + grd_hm_time) * 1000.0,
            zone_ms=(img_zone_time + grd_zone_time) * 1000.0,
            dwell_ms=(img_dwell_time + grd_dwell_time) * 1000.0,
            total_ms=wall_total * 1000.0,
            fps=frames_processed / wall_total if wall_total > 0 else 0.0,
        )

        # Image Spatial Diagnostics
        image_spatial = SpatialDiagnostics(
            total_detections=len(all_detections),
            total_track_observations=len(all_tracks),
            total_footpoints=len(all_footpoints),
            valid_footpoints=len(all_footpoints),
            invalid_footpoints=img_heatmap.invalid_points,
            out_of_bounds_footpoints=img_heatmap.out_of_bounds_points,
            total_trajectories=len(img_trajectories),
            trajectories_with_gaps=traj_stats.trajectories_with_gaps,
            total_gaps=traj_stats.total_gaps,
        )

        # Image Heatmap Diagnostics
        img_occupied = int(np.count_nonzero(img_heatmap_data.raw_counts))
        img_total_cells = int(img_heatmap_data.raw_counts.size)
        image_heatmap_diag = HeatmapDiagnostics(
            total_accumulated=img_heatmap.total_points,
            grid_shape=img_heatmap_data.shape,
            occupied_cells=img_occupied,
            occupancy_ratio=img_occupied / img_total_cells if img_total_cells > 0 else 0.0,
            max_cell_count=img_heatmap_data.max_count,
            out_of_bounds_points=img_heatmap.out_of_bounds_points,
            invalid_points=img_heatmap.invalid_points,
        )

        # Image Zone Diagnostics
        img_in_zone = sum(1 for m in all_img_memberships if m.is_in_zone)
        img_single = sum(1 for m in all_img_memberships if len(m.zone_ids) == 1)
        img_multi = sum(1 for m in all_img_memberships if len(m.zone_ids) > 1)
        img_outside = sum(1 for m in all_img_memberships if len(m.zone_ids) == 0)
        img_unique_visitors = {
            z.zone_id: len(set(v.track_id for v in img_dwell_engine.get_visits_for_zone(z.zone_id)))
            for z in self.config.image_zones
        }
        image_zone_diag = ZoneDiagnostics(
            total_memberships=len(all_img_memberships),
            inside_at_least_one_zone=img_in_zone,
            outside_all_zones=img_outside,
            single_zone_observations=img_single,
            multi_zone_observations=img_multi,
            unique_visitors_per_zone=img_unique_visitors,
        )

        # Image Dwell Diagnostics
        img_visits_by_zone = {
            z.zone_id: len(img_dwell_engine.get_visits_for_zone(z.zone_id))
            for z in self.config.image_zones
        }
        img_dwell_s = sum(v.duration_seconds for v in img_visits)
        image_dwell_diag = DwellDiagnostics(
            total_visits=len(img_visits),
            unique_visitors=len(set(v.track_id for v in img_visits)),
            total_dwell_seconds=img_dwell_s,
            average_dwell_seconds=img_dwell_s / len(img_visits) if img_visits else 0.0,
            max_dwell_seconds=max((v.duration_seconds for v in img_visits), default=0.0),
            visits_by_zone=img_visits_by_zone,
        )

        # Ground Projection Diagnostics
        val_in_roi = sum(1 for o in all_grd_observations if o.is_valid and not o.is_extrapolated)
        val_extrap = sum(1 for o in all_grd_observations if o.is_valid and o.is_extrapolated)
        invalid_grd = sum(1 for o in all_grd_observations if not o.is_valid)

        err_codes: Dict[str, int] = defaultdict(int)
        for o in all_grd_observations:
            if o.error_code:
                err_codes[o.error_code] += 1

        total_grd_accepted = sum(t.length for t in grd_trajectories.values())
        total_grd_excluded = grd_traj_builder.invalid_ignored_count + grd_traj_builder.extrapolated_excluded_count

        grd_proj_diag = GroundProjectionDiagnostics(
            total_projected=len(all_grd_observations),
            valid_in_roi=val_in_roi,
            valid_extrapolated=val_extrap,
            invalid_projections=invalid_grd,
            analytics_accepted=total_grd_accepted,
            analytics_excluded=total_grd_excluded,
            error_codes=dict(err_codes),
        )

        # Ground Zone tallies
        grd_in_zone = sum(1 for m in all_grd_memberships if m.is_in_zone)
        grd_outside = sum(1 for m in all_grd_memberships if not m.is_in_zone)
        grd_single = sum(1 for m in all_grd_memberships if len(m.zone_ids) == 1)
        grd_multi = sum(1 for m in all_grd_memberships if len(m.zone_ids) > 1)

        # Ground Dwell tallies
        grd_visits_by_zone = {
            z.zone_id: len(grd_dwell_engine.get_visits_for_zone(z.zone_id))
            for z in self.config.ground_zones
        }
        grd_dwell_s = sum(v.duration_seconds for v in grd_visits)

        # Audit Conservation Equations (AC-03 to AC-07)
        audit = self._audit_conservation(
            all_footpoints=all_footpoints,
            all_grd_observations=all_grd_observations,
            img_heatmap=img_heatmap,
            grd_heatmap=grd_heatmap,
            img_zone_diag=image_zone_diag,
            grd_in_zone=grd_in_zone,
            grd_outside=grd_outside,
            all_img_memberships=all_img_memberships,
            all_grd_memberships=all_grd_memberships,
            img_visits=img_visits,
            grd_visits=grd_visits,
            val_in_roi=val_in_roi,
            val_extrap=val_extrap,
            invalid_grd=invalid_grd,
        )

        return FullSequenceEvaluationReport(
            sequence_name=sequence.info.name,
            frames_processed=frames_processed,
            total_sequence_frames=sequence.info.seq_length,
            fps=self.config.fps,
            resolution=(self.config.image_width, self.config.image_height),
            detection_metrics=detection_metrics,
            tracking_metrics=tracking_metrics,
            trajectory_stats=traj_stats,
            image_spatial=image_spatial,
            image_heatmap=image_heatmap_diag,
            image_zone=image_zone_diag,
            image_dwell=image_dwell_diag,
            ground_projection=grd_proj_diag,
            ground_heatmap_accumulated=grd_heatmap.total_accumulated,
            ground_heatmap_out_of_bounds=grd_heatmap.out_of_bounds_points,
            ground_heatmap_invalid=grd_heatmap.invalid_points,
            ground_heatmap_extrap_rejected=grd_heatmap.extrapolated_rejected_points,
            ground_heatmap_occupied_cells=int(np.count_nonzero(grd_heatmap_data.raw_counts)),
            ground_heatmap_max_cell_count=grd_heatmap_data.max_cell_count,
            ground_zone_total_memberships=len(all_grd_memberships),
            ground_zone_inside=grd_in_zone,
            ground_zone_outside=grd_outside,
            ground_zone_single=grd_single,
            ground_zone_multi=grd_multi,
            ground_dwell_total_visits=len(grd_visits),
            ground_dwell_unique_visitors=len(set(v.track_id for v in grd_visits)),
            ground_dwell_total_seconds=grd_dwell_s,
            ground_dwell_average_seconds=grd_dwell_s / len(grd_visits) if grd_visits else 0.0,
            ground_dwell_visits_by_zone=grd_visits_by_zone,
            cross_space_comparison=comparison_report,
            conservation_audit=audit,
            timing=stage_timing,
            peak_rss_mb=peak_rss_mb,
        )

    def _compute_trajectory_stats(
        self,
        trajectories: Dict[int, Trajectory],
        fps: float,
    ) -> TrajectorySummaryStats:
        """Compute full-sequence trajectory gap and lifespan statistics."""
        if not trajectories:
            return TrajectorySummaryStats(
                total_trajectories=0,
                total_observations=0,
                mean_observations_per_track=0.0,
                trajectories_with_gaps=0,
                total_gaps=0,
                min_gap_frames=0,
                max_gap_frames=0,
                mean_gap_frames=0.0,
                median_gap_frames=0.0,
                min_lifespan_frames=0,
                max_lifespan_frames=0,
                mean_lifespan_frames=0.0,
                median_lifespan_frames=0.0,
                min_lifespan_seconds=0.0,
                max_lifespan_seconds=0.0,
                mean_lifespan_seconds=0.0,
                median_lifespan_seconds=0.0,
                frame_coverage_ratio=0.0,
            )

        total_obs = sum(len(t) for t in trajectories.values())
        trajs_with_gaps = sum(1 for t in trajectories.values() if t.has_gaps)

        all_gap_lengths: List[int] = []
        lifespans_frames: List[int] = []

        for t in trajectories.values():
            if len(t) == 0:
                continue
            lifespan = t.points[-1].frame_index - t.points[0].frame_index + 1
            lifespans_frames.append(lifespan)

            for i in range(len(t.points) - 1):
                gap = t.points[i + 1].frame_index - t.points[i].frame_index - 1
                if gap > 0:
                    all_gap_lengths.append(gap)

        total_gaps = len(all_gap_lengths)
        min_gap = min(all_gap_lengths) if all_gap_lengths else 0
        max_gap = max(all_gap_lengths) if all_gap_lengths else 0
        mean_gap = statistics.mean(all_gap_lengths) if all_gap_lengths else 0.0
        median_gap = float(statistics.median(all_gap_lengths)) if all_gap_lengths else 0.0

        min_ls = min(lifespans_frames) if lifespans_frames else 0
        max_ls = max(lifespans_frames) if lifespans_frames else 0
        mean_ls = statistics.mean(lifespans_frames) if lifespans_frames else 0.0
        median_ls = float(statistics.median(lifespans_frames)) if lifespans_frames else 0.0

        fps_safe = fps if fps > 0 else 30.0
        sum_lifespans = sum(lifespans_frames)
        coverage_ratio = total_obs / sum_lifespans if sum_lifespans > 0 else 1.0

        return TrajectorySummaryStats(
            total_trajectories=len(trajectories),
            total_observations=total_obs,
            mean_observations_per_track=total_obs / len(trajectories),
            trajectories_with_gaps=trajs_with_gaps,
            total_gaps=total_gaps,
            min_gap_frames=min_gap,
            max_gap_frames=max_gap,
            mean_gap_frames=mean_gap,
            median_gap_frames=median_gap,
            min_lifespan_frames=min_ls,
            max_lifespan_frames=max_ls,
            mean_lifespan_frames=mean_ls,
            median_lifespan_frames=median_ls,
            min_lifespan_seconds=min_ls / fps_safe,
            max_lifespan_seconds=max_ls / fps_safe,
            mean_lifespan_seconds=mean_ls / fps_safe,
            median_lifespan_seconds=median_ls / fps_safe,
            frame_coverage_ratio=coverage_ratio,
        )

    def _audit_conservation(
        self,
        all_footpoints: List[FootpointObservation],
        all_grd_observations: List[GroundObservation],
        img_heatmap: HeatmapAccumulator,
        grd_heatmap: GroundHeatmapAccumulator,
        img_zone_diag: ZoneDiagnostics,
        grd_in_zone: int,
        grd_outside: int,
        all_img_memberships: List[ZoneMembership],
        all_grd_memberships: List[GroundZoneMembership],
        img_visits: Sequence[ZoneVisit],
        grd_visits: Sequence[ZoneVisit],
        val_in_roi: int,
        val_extrap: int,
        invalid_grd: int,
    ) -> FullConservationAudit:
        """Audit all 7 conservation equations with exact integer equality."""
        # AC-03: Projection State Conservation
        n_proj = len(all_grd_observations)
        n_proj_sum = val_in_roi + val_extrap + invalid_grd
        chk_proj = ConservationCheck(
            name="AC-03: Projection State Conservation",
            passed=(n_proj == n_proj_sum),
            left_value=n_proj,
            right_value=n_proj_sum,
            details=f"N_projected ({n_proj}) == In-ROI ({val_in_roi}) + Extrapolated ({val_extrap}) + Invalid ({invalid_grd})",
        )

        # AC-04: Image Heatmap Conservation
        n_fp = len(all_footpoints)
        img_hm_sum = img_heatmap.total_points + img_heatmap.out_of_bounds_points + img_heatmap.invalid_points
        chk_img_hm = ConservationCheck(
            name="AC-04: Image Heatmap Conservation",
            passed=(n_fp == img_hm_sum),
            left_value=n_fp,
            right_value=img_hm_sum,
            details=f"N_footpoints ({n_fp}) == Accumulated ({img_heatmap.total_points}) + OOB ({img_heatmap.out_of_bounds_points}) + Invalid ({img_heatmap.invalid_points})",
        )

        # AC-05: Ground Heatmap Conservation
        n_grd_in = len(all_grd_observations)
        grd_hm_sum = (
            grd_heatmap.total_accumulated
            + grd_heatmap.out_of_bounds_points
            + grd_heatmap.invalid_points
            + grd_heatmap.extrapolated_rejected_points
        )
        chk_grd_hm = ConservationCheck(
            name="AC-05: Ground Heatmap Conservation",
            passed=(n_grd_in == grd_hm_sum),
            left_value=n_grd_in,
            right_value=grd_hm_sum,
            details=f"N_ground_input ({n_grd_in}) == Accumulated ({grd_heatmap.total_accumulated}) + OOB ({grd_heatmap.out_of_bounds_points}) + Invalid ({grd_heatmap.invalid_points}) + ExtrapRejected ({grd_heatmap.extrapolated_rejected_points})",
        )

        # AC-06a: Image Zone Conservation
        img_zone_sum = img_zone_diag.inside_at_least_one_zone + img_zone_diag.outside_all_zones
        chk_img_zone = ConservationCheck(
            name="AC-06a: Image Zone Conservation",
            passed=(n_fp == img_zone_sum),
            left_value=n_fp,
            right_value=img_zone_sum,
            details=f"N_footpoints ({n_fp}) == Inside ({img_zone_diag.inside_at_least_one_zone}) + Outside ({img_zone_diag.outside_all_zones})",
        )

        # AC-06b: Ground Zone Conservation
        grd_zone_sum = grd_in_zone + grd_outside
        chk_grd_zone = ConservationCheck(
            name="AC-06b: Ground Zone Conservation",
            passed=(n_grd_in == grd_zone_sum),
            left_value=n_grd_in,
            right_value=grd_zone_sum,
            details=f"N_ground_obs ({n_grd_in}) == Inside ({grd_in_zone}) + Outside ({grd_outside})",
        )

        # AC-07a: Image Dwell Overlap Conservation
        total_img_dwell_obs = sum(v.observation_count for v in img_visits)
        img_membership_assignments = sum(len(m.zone_ids) for m in all_img_memberships)
        chk_img_dwell = ConservationCheck(
            name="AC-07a: Image Dwell Overlap Conservation",
            passed=(total_img_dwell_obs == img_membership_assignments),
            left_value=total_img_dwell_obs,
            right_value=img_membership_assignments,
            details=f"Total Dwell Observations ({total_img_dwell_obs}) == Total Zone Assignments ({img_membership_assignments})",
        )

        # AC-07b: Ground Dwell Overlap Conservation
        total_grd_dwell_obs = sum(v.observation_count for v in grd_visits)
        grd_membership_assignments = sum(len(m.zone_ids) for m in all_grd_memberships)
        chk_grd_dwell = ConservationCheck(
            name="AC-07b: Ground Dwell Overlap Conservation",
            passed=(total_grd_dwell_obs == grd_membership_assignments),
            left_value=total_grd_dwell_obs,
            right_value=grd_membership_assignments,
            details=f"Total Ground Dwell Observations ({total_grd_dwell_obs}) == Total Ground Zone Assignments ({grd_membership_assignments})",
        )

        return FullConservationAudit(
            projection_conservation=chk_proj,
            image_heatmap_conservation=chk_img_hm,
            ground_heatmap_conservation=chk_grd_hm,
            image_zone_conservation=chk_img_zone,
            ground_zone_conservation=chk_grd_zone,
            image_dwell_conservation=chk_img_dwell,
            ground_dwell_conservation=chk_grd_dwell,
        )

    def _evaluate_detection_metrics(
        self,
        detections: List[PersonDetection],
        ground_truth: List[Any],
        iou_threshold: float = 0.5,
    ) -> DetectionMetrics:
        """Evaluate detections against active pedestrians using frozen Phase 2 contract."""
        dets_by_frame: Dict[int, List[PersonDetection]] = defaultdict(list)
        for d in detections:
            dets_by_frame[d.frame_index].append(d)

        # Active pedestrian ground truth (class_id == 1 and conf == 1.0)
        gt_by_frame: Dict[int, List[Any]] = defaultdict(list)
        for g in ground_truth:
            if g.is_pedestrian and g.is_active:
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

    def _get_peak_rss_mb(self) -> float:
        """Sample current process resident memory in megabytes (observational)."""
        if psutil is not None:
            try:
                proc = psutil.Process(os.getpid())
                return proc.memory_info().rss / (1024.0 * 1024.0)
            except Exception:
                return 0.0
        return 0.0
