"""Visualization tools for ground-plane movement analytics."""

from typing import Dict, Optional, Sequence, Tuple
import cv2
import numpy as np

from people_analytics.ground_analytics.models import (
    GroundHeatmapData,
    GroundObservation,
    GroundTrajectory,
    GroundZone,
)


def draw_ground_analytics_map(
    zones: Sequence[GroundZone] = (),
    trajectories: Optional[Dict[int, GroundTrajectory]] = None,
    active_observations: Sequence[GroundObservation] = (),
    heatmap_data: Optional[GroundHeatmapData] = None,
    canvas_size: Tuple[int, int] = (800, 800),
    margin: int = 60,
    background_color: Tuple[int, int, int] = (24, 24, 28),
) -> np.ndarray:
    """Render a comprehensive bird's-eye ground plane visualization.

    Includes zones, trajectories, active observations, and optional planar heatmap.

    Args:
        zones: Sequence of GroundZone instances.
        trajectories: Optional dict mapping track_id to GroundTrajectory.
        active_observations: Sequence of GroundObservation instances for current frame.
        heatmap_data: Optional GroundHeatmapData to render as background density.
        canvas_size: (width, height) of output canvas in pixels.
        margin: Padding around coordinate grid.
        background_color: Canvas background color (BGR).

    Returns:
        Rendered BGR image array (canvas_height, canvas_width, 3).
    """
    canvas_w, canvas_h = canvas_size
    canvas = np.full((canvas_h, canvas_w, 3), background_color, dtype=np.uint8)

    # 1. Determine ground bounds
    all_x: list[float] = []
    all_y: list[float] = []

    for z in zones:
        all_x.extend([vx for vx, _ in z.vertices])
        all_y.extend([vy for _, vy in z.vertices])

    for obs in active_observations:
        if obs.is_valid:
            all_x.append(obs.x)
            all_y.append(obs.y)

    if trajectories:
        for t in trajectories.values():
            all_x.extend([p.x for p in t.points])
            all_y.extend([p.y for p in t.points])

    if heatmap_data is not None:
        all_x.extend([heatmap_data.config.min_x, heatmap_data.config.max_x])
        all_y.extend([heatmap_data.config.min_y, heatmap_data.config.max_y])

    if not all_x:
        all_x = [0.0, 20.0]
        all_y = [0.0, 10.0]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    min_x -= 0.1 * span_x
    max_x += 0.1 * span_x
    min_y -= 0.1 * span_y
    max_y += 0.1 * span_y

    plot_w = canvas_w - 2 * margin
    plot_h = canvas_h - 2 * margin

    def to_canvas(gx: float, gy: float) -> Tuple[int, int]:
        u = int(margin + (gx - min_x) / (max_x - min_x) * plot_w)
        v = int(canvas_h - margin - (gy - min_y) / (max_y - min_y) * plot_h)
        return (u, v)

    # 2. Draw Planar Heatmap if provided
    if heatmap_data is not None and heatmap_data.max_cell_count > 0:
        cfg = heatmap_data.config
        norm_grid = heatmap_data.raw_counts / heatmap_data.max_cell_count
        u_heat = (norm_grid * 255).astype(np.uint8)
        color_heat = cv2.applyColorMap(u_heat, cv2.COLORMAP_JET)

        # Rescale heatmap to plot area
        u0, v0 = to_canvas(cfg.min_x, cfg.max_y)
        u1, v1 = to_canvas(cfg.max_x, cfg.min_y)
        target_w = max(u1 - u0, 1)
        target_h = max(v1 - v0, 1)

        heat_resized = cv2.resize(color_heat, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Blend into canvas
        sub_canvas = canvas[v0 : v0 + target_h, u0 : u0 + target_w]
        if sub_canvas.shape[:2] == heat_resized.shape[:2]:
            cv2.addWeighted(sub_canvas, 0.6, heat_resized, 0.4, 0, sub_canvas)

    # 3. Draw Grid and Axes
    grid_color = (45, 45, 52)
    axis_color = (90, 90, 105)
    num_ticks = 5

    for step in range(num_ticks + 1):
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

    cv2.rectangle(canvas, (margin, margin), (canvas_w - margin, canvas_h - margin), axis_color, 1)

    # 4. Draw Ground Zones
    zone_colors = [
        (60, 140, 60),
        (160, 100, 40),
        (140, 60, 140),
        (50, 120, 180),
    ]

    overlay = canvas.copy()
    for i, z in enumerate(zones):
        poly = np.array([to_canvas(vx, vy) for vx, vy in z.vertices], dtype=np.int32)
        color = zone_colors[i % len(zone_colors)]
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(canvas, [poly], isClosed=True, color=(200, 230, 200), thickness=2)

        # Zone label
        c_x = float(np.mean([vx for vx, _ in z.vertices]))
        c_y = float(np.mean([vy for _, vy in z.vertices]))
        lu, lv = to_canvas(c_x, c_y)
        cv2.putText(
            canvas,
            z.name or z.zone_id,
            (lu - 20, lv),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    # 5. Draw Ground Trajectories
    traj_palette = [
        (255, 140, 0),
        (0, 255, 140),
        (255, 0, 140),
        (140, 255, 0),
        (0, 140, 255),
    ]

    if trajectories:
        for tid, traj in trajectories.items():
            if traj.length < 2:
                continue
            color = traj_palette[tid % len(traj_palette)]
            pts_canvas = [to_canvas(p.x, p.y) for p in traj.points]
            for i in range(len(pts_canvas) - 1):
                cv2.line(canvas, pts_canvas[i], pts_canvas[i + 1], color, 2, cv2.LINE_AA)

    # 6. Draw Active Observations
    for obs in active_observations:
        if not obs.is_valid:
            continue
        cu, cv = to_canvas(obs.x, obs.y)
        color = traj_palette[obs.track_id % len(traj_palette)]
        if obs.is_extrapolated:
            cv2.circle(canvas, (cu, cv), 7, (0, 165, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (cu, cv), 3, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (cu, cv), 5, color, -1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            f"T:{obs.track_id}",
            (cu + 7, cv - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    # 7. Header
    title = "Ground-Plane Analytics Map [ARBITRARY_PLANAR]"
    cv2.putText(canvas, title, (margin, margin - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    return canvas


def draw_dual_view_diagnostics(
    image_view: np.ndarray,
    ground_view: np.ndarray,
) -> np.ndarray:
    """Create a side-by-side synchronized diagnostic canvas of image view and ground view.

    Args:
        image_view: Perspective camera frame (H1, W1, 3).
        ground_view: Bird's-eye ground map (H2, W2, 3).

    Returns:
        Side-by-side composite image.
    """
    target_h = max(image_view.shape[0], ground_view.shape[0])

    # Resize proportionally to match height
    def match_height(img: np.ndarray, h: int) -> np.ndarray:
        if img.shape[0] == h:
            return img
        scale = h / img.shape[0]
        w = int(img.shape[1] * scale)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    left = match_height(image_view, target_h)
    right = match_height(ground_view, target_h)

    divider = np.full((target_h, 4, 3), (80, 80, 90), dtype=np.uint8)
    return np.hstack([left, divider, right])
