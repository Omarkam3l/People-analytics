"""Image-space heatmap package."""

from people_analytics.heatmap.accumulator import HeatmapAccumulator
from people_analytics.heatmap.models import HeatmapConfig, HeatmapData
from people_analytics.heatmap.visualization import overlay_heatmap, render_heatmap

__all__ = [
    "HeatmapAccumulator",
    "HeatmapConfig",
    "HeatmapData",
    "overlay_heatmap",
    "render_heatmap",
]
