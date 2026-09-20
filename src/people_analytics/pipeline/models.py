"""Telemetry and report data models for end-to-end pipeline evaluation."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from people_analytics.evaluation.detection_metrics import DetectionMetrics
from people_analytics.evaluation.tracking_metrics import TrackingMetrics


@dataclass(frozen=True)
class StageTiming:
    """High-resolution latency measurements per pipeline stage in milliseconds."""
    detection_ms: float
    tracking_ms: float
    footpoint_ms: float
    trajectory_ms: float
    heatmap_ms: float
    zone_ms: float
    dwell_ms: float
    total_ms: float
    fps: float


@dataclass(frozen=True)
class SpatialDiagnostics:
    """Diagnostics auditing spatial objects and coordinate conservation across stages."""
    total_detections: int
    total_track_observations: int
    total_footpoints: int
    valid_footpoints: int
    invalid_footpoints: int
    out_of_bounds_footpoints: int
    total_trajectories: int
    trajectories_with_gaps: int
    total_gaps: int


@dataclass(frozen=True)
class HeatmapDiagnostics:
    """Diagnostics auditing image-space discrete heatmap accumulation."""
    total_accumulated: int
    grid_shape: Tuple[int, int]
    occupied_cells: int
    occupancy_ratio: float
    max_cell_count: float
    out_of_bounds_points: int
    invalid_points: int


@dataclass(frozen=True)
class ZoneDiagnostics:
    """Diagnostics auditing spatial zone membership queries."""
    total_memberships: int
    inside_at_least_one_zone: int
    outside_all_zones: int
    single_zone_observations: int
    multi_zone_observations: int
    unique_visitors_per_zone: Dict[str, int]

    @property
    def in_zone_observations(self) -> int:
        """Alias for observations inside at least one zone."""
        return self.inside_at_least_one_zone

    @property
    def outside_observations(self) -> int:
        """Alias for observations outside all zones."""
        return self.outside_all_zones


@dataclass(frozen=True)
class DwellDiagnostics:
    """Diagnostics auditing dwell intervals and visit analytics."""
    total_visits: int
    unique_visitors: int
    total_dwell_seconds: float
    average_dwell_seconds: float
    max_dwell_seconds: float
    visits_by_zone: Dict[str, int]


@dataclass(frozen=True)
class InvariantCheckResult:
    """Status and details of an individual integration invariant audit."""
    passed: bool
    invariant_name: str
    details: str


@dataclass(frozen=True)
class EndToEndReport:
    """Unified evaluation and engineering audit report for an analyzed sequence."""
    sequence_name: str
    frames_processed: int
    detection_metrics: Optional[DetectionMetrics]
    tracking_metrics: Optional[TrackingMetrics]
    timing: StageTiming
    spatial: SpatialDiagnostics
    heatmap: HeatmapDiagnostics
    zone: ZoneDiagnostics
    dwell: DwellDiagnostics
    invariants: Tuple[InvariantCheckResult, ...] = ()

    @property
    def all_invariants_passed(self) -> bool:
        """True if every audited integration invariant passed."""
        return all(inv.passed for inv in self.invariants) if self.invariants else True
