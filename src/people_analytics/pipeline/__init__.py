"""End-to-end pipeline execution and audit module."""

from people_analytics.pipeline.diagnostics import draw_pipeline_diagnostics
from people_analytics.pipeline.models import (
    DwellDiagnostics,
    EndToEndReport,
    HeatmapDiagnostics,
    InvariantCheckResult,
    SpatialDiagnostics,
    StageTiming,
    ZoneDiagnostics,
)
from people_analytics.pipeline.runner import PipelineRunner

__all__ = [
    "StageTiming",
    "SpatialDiagnostics",
    "HeatmapDiagnostics",
    "ZoneDiagnostics",
    "DwellDiagnostics",
    "InvariantCheckResult",
    "EndToEndReport",
    "PipelineRunner",
    "draw_pipeline_diagnostics",
]
