"""Visualization utilities for spatial zones."""

from typing import Optional, Sequence, Tuple
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from people_analytics.zone.models import Zone, ZoneMembership

# Curated palette for distinct zone rendering
_ZONE_PALETTE = [
    (255, 100, 0),   # Blue/Azure
    (0, 180, 255),   # Orange
    (0, 220, 0),     # Green
    (180, 0, 255),   # Magenta/Violet
    (0, 255, 255),   # Yellow
    (255, 0, 180),   # Purple
]


def _get_zone_color(index: int) -> Tuple[int, int, int]:
    return _ZONE_PALETTE[index % len(_ZONE_PALETTE)]


def draw_zones(
    image: np.ndarray,
    zones: Sequence[Zone],
    memberships: Optional[Sequence[ZoneMembership]] = None,
    alpha: float = 0.25,
) -> np.ndarray:
    """Render spatial zones and optional footpoint membership markers onto an image canvas.

    Args:
        image: Source BGR image canvas (H, W, 3).
        zones: Sequence of Zone objects to draw.
        memberships: Optional sequence of ZoneMembership instances to render as markers.
        alpha: Opacity of filled zone polygons in [0.0, 1.0] (default: 0.25).

    Returns:
        New annotated image frame (does not mutate input image).
    """
    output = image.copy()
    if cv2 is None or len(zones) == 0:
        return output

    overlay = output.copy()

    for idx, zone in enumerate(zones):
        color = _get_zone_color(idx)
        pts = np.array([[int(round(x)), int(round(y))] for x, y in zone.vertices], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Fill zone polygon on overlay
        cv2.fillPoly(overlay, [pts], color)
        # Draw solid boundary outline on output
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Draw label near first vertex or centroid
        label = zone.name if zone.name else zone.zone_id
        min_x = int(np.min(pts[:, 0, 0]))
        min_y = int(np.min(pts[:, 0, 1]))
        cv2.putText(
            output,
            label,
            (min_x + 5, max(min_y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (min_x + 5, max(min_y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    # Blend transparent zone fill
    cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0, output)

    # Optionally draw footpoint markers
    if memberships is not None:
        for m in memberships:
            px = int(round(m.x))
            py = int(round(m.y))
            # Green if in a zone, Gray if outside
            pt_color = (0, 255, 0) if m.is_in_zone else (160, 160, 160)
            cv2.circle(output, (px, py), 4, pt_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(output, (px, py), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return output
