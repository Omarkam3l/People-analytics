"""Minimal visualization utilities for footpoints."""

from typing import List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from people_analytics.footpoint.models import FootpointObservation


def draw_footpoints(
    image: np.ndarray,
    observations: List[FootpointObservation],
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 4,
    show_labels: bool = True,
) -> np.ndarray:
    """Render footpoints onto an image canvas.

    Args:
        image: BGR numpy image array (H, W, 3).
        observations: List of FootpointObservation instances to draw.
        color: Marker color in BGR format (default: Red).
        radius: Radius of the footpoint marker circle in pixels.
        show_labels: Whether to display track_id text above the footpoint.

    Returns:
        Annotated image as a copy of the input image.
    """
    canvas = image.copy()
    if cv2 is None or len(observations) == 0:
        return canvas

    for obs in observations:
        center_x = int(round(obs.x))
        center_y = int(round(obs.y))

        # Draw footpoint marker
        cv2.circle(canvas, (center_x, center_y), radius, color, -1)
        cv2.circle(canvas, (center_x, center_y), radius + 1, (255, 255, 255), 1)

        if show_labels:
            label = f"ID:{obs.track_id}"
            cv2.putText(
                canvas,
                label,
                (center_x + 6, center_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (center_x + 6, center_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    return canvas
