"""Diagnostic visualization for top-down bird's-eye ground plane representations."""

from typing import Dict, Optional, Sequence, Tuple
import cv2
import numpy as np

from people_analytics.ground.models import GroundPlaneCalibration, GroundPoint


def draw_ground_plane_map(
    calibration: GroundPlaneCalibration,
    points: Sequence[GroundPoint] = (),
    trajectories: Optional[Dict[int, Sequence[GroundPoint]]] = None,
    canvas_size: Tuple[int, int] = (800, 800),
    margin: int = 60,
    background_color: Tuple[int, int, int] = (24, 24, 28),
) -> np.ndarray:
    """Render a top-down 2D bird's-eye map displaying calibration ROI and ground points.

    Args:
        calibration: GroundPlaneCalibration defining target frame and reference coordinates.
        points: Optional sequence of active GroundPoint instances to plot.
        trajectories: Optional dict mapping track_id to sequence of GroundPoints.
        canvas_size: (width, height) of output canvas in pixels.
        margin: Padding in pixels around coordinate grid.
        background_color: BGR tuple for canvas background.

    Returns:
        np.ndarray: Rendered BGR image array (canvas_height, canvas_width, 3).
    """
    canvas_w, canvas_h = canvas_size
    canvas = np.full((canvas_h, canvas_w, 3), background_color, dtype=np.uint8)

    # 1. Determine ground-plane bounding box from calibration reference points
    ref_x = [p.ground_x for p in calibration.reference_points]
    ref_y = [p.ground_y for p in calibration.reference_points]

    # Include any valid ground points in bounding box calculation
    valid_pts = [p for p in points if p.is_valid]
    if valid_pts:
        ref_x.extend([p.x for p in valid_pts])
        ref_y.extend([p.y for p in valid_pts])

    min_x, max_x = min(ref_x), max(ref_x)
    min_y, max_y = min(ref_y), max(ref_y)

    # Pad bounds to prevent drawing on the exact edge
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    min_x -= 0.1 * span_x
    max_x += 0.1 * span_x
    min_y -= 0.1 * span_y
    max_y += 0.1 * span_y

    plot_w = canvas_w - 2 * margin
    plot_h = canvas_h - 2 * margin

    def to_canvas(gx: float, gy: float) -> Tuple[int, int]:
        # X maps left-to-right; Y maps bottom-to-top (standard Cartesian top-down)
        u = int(margin + (gx - min_x) / (max_x - min_x) * plot_w)
        v = int(canvas_h - margin - (gy - min_y) / (max_y - min_y) * plot_h)
        return (u, v)

    # 2. Draw Grid and Axes
    grid_color = (45, 45, 52)
    axis_color = (90, 90, 105)
    num_ticks = 5

    for step in range(num_ticks + 1):
        # Vertical grid lines (constant X)
        gx = min_x + step * (max_x - min_x) / num_ticks
        u, _ = to_canvas(gx, min_y)
        cv2.line(canvas, (u, margin), (u, canvas_h - margin), grid_color, 1)
        cv2.putText(
            canvas,
            f"{gx:.1f}",
            (u - 15, canvas_h - margin + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (160, 160, 170),
            1,
            cv2.LINE_AA,
        )

        # Horizontal grid lines (constant Y)
        gy = min_y + step * (max_y - min_y) / num_ticks
        _, v = to_canvas(min_x, gy)
        cv2.line(canvas, (margin, v), (canvas_w - margin, v), grid_color, 1)
        cv2.putText(
            canvas,
            f"{gy:.1f}",
            (10, v + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (160, 160, 170),
            1,
            cv2.LINE_AA,
        )

    # Draw outer plot bounding box
    cv2.rectangle(
        canvas,
        (margin, margin),
        (canvas_w - margin, canvas_h - margin),
        axis_color,
        1,
    )

    # 3. Draw Calibration Reference Polygon
    calib_poly_canvas = np.array(
        [to_canvas(p.ground_x, p.ground_y) for p in calibration.reference_points],
        dtype=np.int32,
    )
    # Semi-transparent overlay for calibration region
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [calib_poly_canvas], (50, 70, 90))
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    cv2.polylines(canvas, [calib_poly_canvas], isClosed=True, color=(100, 160, 220), thickness=2)

    for p in calibration.reference_points:
        cu, cv = to_canvas(p.ground_x, p.ground_y)
        cv2.circle(canvas, (cu, cv), 4, (0, 215, 255), -1)
        if p.label:
            cv2.putText(
                canvas,
                p.label,
                (cu + 6, cv - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 215, 255),
                1,
                cv2.LINE_AA,
            )

    # 4. Draw Trajectories if provided
    color_palette = [
        (255, 128, 0),
        (0, 255, 128),
        (255, 0, 128),
        (128, 255, 0),
        (0, 128, 255),
        (255, 200, 0),
    ]

    if trajectories:
        for tid, tpts in trajectories.items():
            valid_tpts = [p for p in tpts if p.is_valid]
            if len(valid_tpts) < 2:
                continue
            color = color_palette[tid % len(color_palette)]
            pts_canvas = [to_canvas(p.x, p.y) for p in valid_tpts]
            for i in range(len(pts_canvas) - 1):
                cv2.line(canvas, pts_canvas[i], pts_canvas[i + 1], color, 2, cv2.LINE_AA)

    # 5. Draw Active Points
    for pt in valid_pts:
        cu, cv = to_canvas(pt.x, pt.y)
        color = color_palette[pt.track_id % len(color_palette)]
        # Outer ring if extrapolated, solid if inside
        if pt.is_extrapolated:
            cv2.circle(canvas, (cu, cv), 6, (0, 165, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (cu, cv), 3, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (cu, cv), 5, color, -1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            f"ID:{pt.track_id}",
            (cu + 7, cv - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    # 6. Header Information
    units_str = calibration.units
    title = f"Top-Down Ground Plane [{calibration.target_frame.value.upper()}] (Units: {units_str})"
    cv2.putText(
        canvas,
        title,
        (margin, margin - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    subtitle = (
        f"Calib: {calibration.calibration_id} | RMSE: {calibration.reprojection_rmse:.4f} {units_str} | "
        f"Points: {len(valid_pts)}"
    )
    cv2.putText(
        canvas,
        subtitle,
        (margin, margin - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (170, 170, 180),
        1,
        cv2.LINE_AA,
    )

    return canvas
