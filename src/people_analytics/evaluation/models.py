"""Data models and serialization structures for Phase 12 full-system evaluation."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from people_analytics.evaluation.detection_metrics import DetectionMetrics
from people_analytics.evaluation.tracking_metrics import TrackingMetrics
from people_analytics.footpoint.models import FootpointObservation
from people_analytics.ground_analytics.comparison import GroundComparisonReport
from people_analytics.ground_analytics.models import GroundHeatmapData, GroundObservation, GroundZoneMembership
from people_analytics.pipeline.models import (
    DwellDiagnostics,
    HeatmapDiagnostics,
    SpatialDiagnostics,
    StageTiming,
    ZoneDiagnostics,
)
from people_analytics.tracking.models import TrackObservation
from people_analytics.zone.models import ZoneMembership


@dataclass(frozen=True)
class TrajectorySummaryStats:
    """Summary statistics describing full-sequence trajectory construction and gaps."""
    total_trajectories: int
    total_observations: int
    mean_observations_per_track: float
    trajectories_with_gaps: int
    total_gaps: int
    min_gap_frames: int
    max_gap_frames: int
    mean_gap_frames: float
    median_gap_frames: float
    min_lifespan_frames: int
    max_lifespan_frames: int
    mean_lifespan_frames: float
    median_lifespan_frames: float
    min_lifespan_seconds: float
    max_lifespan_seconds: float
    mean_lifespan_seconds: float
    median_lifespan_seconds: float
    frame_coverage_ratio: float


@dataclass(frozen=True)
class GroundProjectionDiagnostics:
    """Detailed diagnostics of homography projection outputs and error provenance."""
    total_projected: int
    valid_in_roi: int
    valid_extrapolated: int
    invalid_projections: int
    analytics_accepted: int
    analytics_excluded: int
    error_codes: Dict[str, int]


@dataclass(frozen=True)
class ConservationCheck:
    """Individual conservation equation verification result."""
    name: str
    passed: bool
    left_value: int
    right_value: int
    details: str


@dataclass(frozen=True)
class FullConservationAudit:
    """Comprehensive suite of all Phase 12 mathematical conservation equations."""
    projection_conservation: ConservationCheck       # AC-03
    image_heatmap_conservation: ConservationCheck    # AC-04
    ground_heatmap_conservation: ConservationCheck   # AC-05
    image_zone_conservation: ConservationCheck       # AC-06 (image)
    ground_zone_conservation: ConservationCheck      # AC-06 (ground)
    image_dwell_conservation: ConservationCheck      # AC-07 (image)
    ground_dwell_conservation: ConservationCheck     # AC-07 (ground)

    @property
    def all_passed(self) -> bool:
        """True if every conservation invariant holds with exact equality."""
        return (
            self.projection_conservation.passed
            and self.image_heatmap_conservation.passed
            and self.ground_heatmap_conservation.passed
            and self.image_zone_conservation.passed
            and self.ground_zone_conservation.passed
            and self.image_dwell_conservation.passed
            and self.ground_dwell_conservation.passed
        )

    @property
    def checks(self) -> Tuple[ConservationCheck, ...]:
        return (
            self.projection_conservation,
            self.image_heatmap_conservation,
            self.ground_heatmap_conservation,
            self.image_zone_conservation,
            self.ground_zone_conservation,
            self.image_dwell_conservation,
            self.ground_dwell_conservation,
        )


@dataclass(frozen=True)
class FullSequenceEvaluationReport:
    """Unified full-system evaluation record for an analyzed MOT17 sequence."""
    sequence_name: str
    frames_processed: int
    total_sequence_frames: int
    fps: float
    resolution: Tuple[int, int]
    detection_metrics: Optional[DetectionMetrics]
    tracking_metrics: Optional[TrackingMetrics]
    trajectory_stats: TrajectorySummaryStats
    image_spatial: SpatialDiagnostics
    image_heatmap: HeatmapDiagnostics
    image_zone: ZoneDiagnostics
    image_dwell: DwellDiagnostics
    ground_projection: GroundProjectionDiagnostics
    ground_heatmap_accumulated: int
    ground_heatmap_out_of_bounds: int
    ground_heatmap_invalid: int
    ground_heatmap_extrap_rejected: int
    ground_heatmap_occupied_cells: int
    ground_heatmap_max_cell_count: float
    ground_zone_total_memberships: int
    ground_zone_inside: int
    ground_zone_outside: int
    ground_zone_single: int
    ground_zone_multi: int
    ground_dwell_total_visits: int
    ground_dwell_unique_visitors: int
    ground_dwell_total_seconds: float
    ground_dwell_average_seconds: float
    ground_dwell_visits_by_zone: Dict[str, int]
    cross_space_comparison: GroundComparisonReport
    conservation_audit: FullConservationAudit
    timing: StageTiming
    peak_rss_mb: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete report into a JSON-compatible dictionary."""
        return {
            "sequence_name": self.sequence_name,
            "frames_processed": self.frames_processed,
            "total_sequence_frames": self.total_sequence_frames,
            "fps": self.fps,
            "resolution": list(self.resolution),
            "detection_metrics": asdict(self.detection_metrics) if self.detection_metrics else None,
            "tracking_metrics": asdict(self.tracking_metrics) if self.tracking_metrics else None,
            "trajectory_stats": asdict(self.trajectory_stats),
            "image_spatial": asdict(self.image_spatial),
            "image_heatmap": {
                **asdict(self.image_heatmap),
                "grid_shape": list(self.image_heatmap.grid_shape),
            },
            "image_zone": asdict(self.image_zone),
            "image_dwell": asdict(self.image_dwell),
            "ground_projection": asdict(self.ground_projection),
            "ground_heatmap": {
                "total_accumulated": self.ground_heatmap_accumulated,
                "out_of_bounds": self.ground_heatmap_out_of_bounds,
                "invalid": self.ground_heatmap_invalid,
                "extrapolated_rejected": self.ground_heatmap_extrap_rejected,
                "occupied_cells": self.ground_heatmap_occupied_cells,
                "max_cell_count": self.ground_heatmap_max_cell_count,
            },
            "ground_zone": {
                "total_memberships": self.ground_zone_total_memberships,
                "inside": self.ground_zone_inside,
                "outside": self.ground_zone_outside,
                "single": self.ground_zone_single,
                "multi": self.ground_zone_multi,
            },
            "ground_dwell": {
                "total_visits": self.ground_dwell_total_visits,
                "unique_visitors": self.ground_dwell_unique_visitors,
                "total_seconds": self.ground_dwell_total_seconds,
                "average_seconds": self.ground_dwell_average_seconds,
                "visits_by_zone": self.ground_dwell_visits_by_zone,
            },
            "cross_space_comparison": asdict(self.cross_space_comparison),
            "conservation_audit": {
                "all_passed": self.conservation_audit.all_passed,
                "checks": [asdict(c) for c in self.conservation_audit.checks],
            },
            "timing": asdict(self.timing),
            "peak_rss_mb": self.peak_rss_mb,
        }


@dataclass
class EvaluationFrameState:
    """State of active tracks, footpoints, and ground observations from the final evaluated frame."""
    frame_index: int
    tracks: List[TrackObservation] = field(default_factory=list)
    footpoints: List[FootpointObservation] = field(default_factory=list)
    image_memberships: List[ZoneMembership] = field(default_factory=list)
    ground_observations: List[GroundObservation] = field(default_factory=list)
    ground_memberships: List[GroundZoneMembership] = field(default_factory=list)
    ground_heatmap_data: Optional[GroundHeatmapData] = None

