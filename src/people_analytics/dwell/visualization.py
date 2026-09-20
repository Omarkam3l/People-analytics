"""Visualization utilities for dwell time and zone visits."""

from typing import Optional, Sequence, Tuple
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from people_analytics.dwell.models import ZoneVisit
from people_analytics.zone.models import Zone

# Palette for distinct zone rendering
_DWELL_PALETTE = [
    (255, 120, 0),   # Azure
    (0, 165, 255),   # Orange
    (0, 210, 0),     # Green
    (190, 0, 240),   # Magenta
    (0, 230, 230),   # Yellow
    (240, 0, 160),   # Purple
]


def _get_color(index: int) -> Tuple[int, int, int]:
    return _DWELL_PALETTE[index % len(_DWELL_PALETTE)]


def draw_dwell_overlay(
    image: np.ndarray,
    zones: Sequence[Zone],
    visits: Sequence[ZoneVisit],
    current_frame: Optional[int] = None,
    alpha: float = 0.25,
) -> np.ndarray:
    """Render zone outlines and dwell statistics overlay onto an image canvas.

    Args:
        image: Source BGR image (H, W, 3).
        zones: Sequence of Zone definitions.
        visits: Sequence of completed or recorded ZoneVisit instances.
        current_frame: Optional video frame index to highlight currently active visits.
        alpha: Opacity for filled polygon overlays (default: 0.25).

    Returns:
        New annotated image frame (does not modify input image in place).
    """
    output = image.copy()
    if cv2 is None or len(zones) == 0:
        return output

    overlay = output.copy()

    # Precompute statistics per zone from provided visits
    zone_stats = {}
    for z in zones:
        z_visits = [v for v in visits if v.zone_id == z.zone_id]
        total_visits = len(z_visits)
        total_dwell = sum(v.duration_seconds for v in z_visits)

        active_count = 0
        if current_frame is not None:
            active_count = sum(
                1 for v in z_visits
                if v.entry_frame <= current_frame <= v.last_observed_frame
            )

        zone_stats[z.zone_id] = {
            "total_visits": total_visits,
            "total_dwell": total_dwell,
            "active_count": active_count,
        }

    for idx, zone in enumerate(zones):
        color = _get_color(idx)
        pts = np.array([[int(round(x)), int(round(y))] for x, y in zone.vertices], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Fill polygon on overlay
        cv2.fillPoly(overlay, [pts], color)
        # Outline boundary
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Calculate bounding centroid for text badge
        cx = int(np.mean([x for x, y in zone.vertices]))
        cy = int(np.mean([y for x, y in zone.vertices]))

        stats = zone_stats[zone.zone_id]
        label_title = f"{zone.name or zone.zone_id}"
        label_stats = f"Visits: {stats['total_visits']} | Dwell: {stats['total_dwell']:.1f}s"
        if current_frame is not None:
            label_stats += f" | Active: {stats['active_count']}"

        # Draw text badges with background box
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1

        (w1, h1), _ = cv2.getTextSize(label_title, font, scale + 0.05, thickness + 1)
        (w2, h2), _ = cv2.getTextSize(label_stats, font, scale, thickness)
        badge_w = max(w1, w2) + 12
        badge_h = h1 + h2 + 16

        bx1 = max(0, cx - badge_w // 2)
        by1 = max(0, cy - badge_h // 2)
        bx2 = min(output.shape[1] - 1, bx1 + badge_w)
        by2 = min(output.shape[0] - 1, by1 + badge_h)

        # Draw semi-opaque dark card for readability
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(output, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

        cv2.putText(
            output,
            label_title,
            (bx1 + 6, by1 + h1 + 4),
            font,
            scale + 0.05,
            (255, 255, 255),
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label_stats,
            (bx1 + 6, by1 + h1 + h2 + 10),
            font,
            scale,
            (200, 200, 200),
            thickness,
            cv2.LINE_AA,
        )

    # Blend overlay with output
    cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, output)
    return output
