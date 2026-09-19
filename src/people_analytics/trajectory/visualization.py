"""Minimal visualization utilities for trajectories."""

from typing import Dict, Optional, Sequence, Tuple
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from people_analytics.trajectory.models import Trajectory

# Curated distinguishable BGR colors for multi-track rendering
_PALETTE = [
    (0, 255, 0),     # Bright Green
    (255, 0, 0),     # Blue
    (0, 200, 255),   # Yellow/Orange
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (255, 128, 0),   # Purple/Indigo
    (0, 128, 255),   # Orange
    (128, 255, 0),   # Lime
]


def _get_track_color(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[(track_id - 1) % len(_PALETTE)]


def draw_trajectories(
    image: np.ndarray,
    trajectories: Sequence[Trajectory],
    current_frame: Optional[int] = None,
    max_trail_frames: Optional[int] = 30,
    gap_threshold: int = 1,
    thickness: int = 2,
) -> np.ndarray:
    """Render movement trajectory polylines onto an image canvas.

    Gap-awareness:
        Consecutive points are only connected by a line if:
            p[i+1].frame_index - p[i].frame_index <= gap_threshold
        This ensures lines are not erroneously drawn across occlusion gaps.

    Args:
        image: BGR numpy image array (H, W, 3).
        trajectories: Sequence of Trajectory objects to render.
        current_frame: Current video frame index. If specified, only points with
            frame_index <= current_frame are drawn.
        max_trail_frames: Maximum historical frame window to draw. If None, draws full history.
        gap_threshold: Maximum frame delta to draw a connecting segment (default: 1).
        thickness: Line thickness in pixels.

    Returns:
        Annotated image as a copy of the input image.
    """
    canvas = image.copy()
    if cv2 is None or len(trajectories) == 0:
        return canvas

    for traj in trajectories:
        if traj.is_empty:
            continue

        color = _get_track_color(traj.track_id)

        # Filter points by frame window
        valid_points = []
        for pt in traj.points:
            if current_frame is not None:
                if pt.frame_index > current_frame:
                    continue
                if max_trail_frames is not None and (current_frame - pt.frame_index) > max_trail_frames:
                    continue
            valid_points.append(pt)

        if len(valid_points) < 2:
            continue

        # Draw line segments, respecting gaps
        for i in range(len(valid_points) - 1):
            p1 = valid_points[i]
            p2 = valid_points[i + 1]

            if (p2.frame_index - p1.frame_index) <= gap_threshold:
                pt1 = (int(round(p1.x)), int(round(p1.y)))
                pt2 = (int(round(p2.x)), int(round(p2.y)))
                cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    return canvas
