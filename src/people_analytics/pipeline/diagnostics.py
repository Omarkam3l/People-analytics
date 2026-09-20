"""Composite diagnostic visualizer rendering multi-layer pipeline analytics onto video frames."""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from people_analytics.dwell.models import ZoneAnalytics, ZoneVisit
from people_analytics.footpoint.models import FootpointObservation
from people_analytics.tracking.models import TrackObservation
from people_analytics.trajectory.models import Trajectory
from people_analytics.zone.models import Zone, ZoneMembership

# Curated high-contrast color palette for identities
_TRACK_PALETTE = [
    (0, 255, 0),     # Bright Green
    (255, 128, 0),   # Blue-Orange
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 165, 255),   # Amber
    (255, 255, 0),   # Cyan
    (180, 105, 255), # Hot Pink
    (144, 238, 144), # Light Green
    (230, 216, 173), # Light Blue
    (180, 238, 180), # Pale Green
]


def _get_track_color(track_id: int) -> Tuple[int, int, int]:
    return _TRACK_PALETTE[abs(track_id) % len(_TRACK_PALETTE)]


def draw_pipeline_diagnostics(
    image: np.ndarray,
    tracks: Sequence[TrackObservation],
    footpoints: Sequence[FootpointObservation],
    zones: Sequence[Zone],
    memberships: Sequence[ZoneMembership],
    trajectories: Optional[Dict[int, Trajectory]] = None,
    zone_analytics: Optional[Dict[str, ZoneAnalytics]] = None,
    frame_index: int = 1,
    throughput_fps: Optional[float] = None,
    alpha: float = 0.25,
) -> np.ndarray:
    """Render a comprehensive diagnostic overlay displaying all pipeline stages simultaneously.

    Layers rendered:
        1. Zones: Semi-transparent polygon fills and boundary outlines.
        2. Trajectories: Historical motion breadcrumb tails (last 30 frames) for active tracks.
        3. Person Bounding Boxes: High-contrast bounding boxes with track ID labels.
        4. Footpoints: Contact point markers on image canvas.
        5. In-Zone Indicators: Highlights for tracks currently inside one or more spatial zones.
        6. Diagnostic HUD: Header banner showing frame index, active tracks, in-zone count, FPS.

    Args:
        image: Source BGR image canvas (H, W, 3).
        tracks: TrackObservation records for the current frame.
        footpoints: FootpointObservation records for the current frame.
        zones: Registered spatial Zone definitions.
        memberships: ZoneMembership records for the current frame.
        trajectories: Optional complete Trajectory lookup dictionary.
        zone_analytics: Optional cumulative ZoneAnalytics lookup.
        frame_index: Current video frame index.
        throughput_fps: Optional measured real-time processing FPS.
        alpha: Opacity for filled polygon overlays.

    Returns:
        New annotated image frame (does not mutate input image in place).
    """
    output = image.copy()
    if cv2 is None:
        return output

    h, w = output.shape[:2]
    overlay = output.copy()

    # 1. Render Zones
    for idx, zone in enumerate(zones):
        color = _get_track_color(idx + 3)
        pts = np.array([[int(round(x)), int(round(y))] for x, y in zone.vertices], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Centroid badge
        cx = int(np.mean([x for x, y in zone.vertices]))
        cy = int(np.mean([y for x, y in zone.vertices]))

        active_in_zone = sum(1 for m in memberships if zone.zone_id in m.zone_ids)
        badge_text = f"Zone {zone.name or zone.zone_id}: {active_in_zone} active"
        if zone_analytics and zone.zone_id in zone_analytics:
            za = zone_analytics[zone.zone_id]
            badge_text += f" | {za.total_visits} visits ({za.total_dwell_time:.1f}s)"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        (tw, th), _ = cv2.getTextSize(badge_text, font, scale, 1)
        bx1 = max(0, cx - tw // 2 - 4)
        by1 = max(0, cy - th // 2 - 4)
        bx2 = min(w - 1, bx1 + tw + 8)
        by2 = min(h - 1, by1 + th + 8)

        cv2.rectangle(output, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(output, (bx1, by1), (bx2, by2), color, 1)
        cv2.putText(output, badge_text, (bx1 + 4, by1 + th + 2), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend zone overlay
    cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, output)

    # Build lookup for active memberships
    membership_by_track = {m.track_id: m for m in memberships}

    # 2. Render Trajectory Trails
    if trajectories:
        for track in tracks:
            tid = track.track_id
            if tid in trajectories:
                traj = trajectories[tid]
                # Slice recent points within past 30 frames
                recent_pts = [
                    (int(round(p.x)), int(round(p.y)))
                    for p in traj.points
                    if 0 <= frame_index - p.frame_index <= 30
                ]
                color = _get_track_color(tid)
                for i in range(len(recent_pts) - 1):
                    p1 = recent_pts[i]
                    p2 = recent_pts[i + 1]
                    cv2.line(output, p1, p2, color, 2, cv2.LINE_AA)

    # 3. Render Person Bounding Boxes & Labels
    for track in tracks:
        tid = track.track_id
        color = _get_track_color(tid)
        x1, y1, x2, y2 = [int(round(v)) for v in track.bbox_xyxy]

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # ID Label
        m = membership_by_track.get(tid)
        in_zones_str = f" [{','.join(m.zone_ids)}]" if m and m.zone_ids else ""
        label = f"ID {tid}{in_zones_str}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        (lw, lh), _ = cv2.getTextSize(label, font, scale, 1)
        cv2.rectangle(output, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, max(lh + 6, y1)), color, -1)
        cv2.putText(output, label, (x1 + 3, max(lh, y1 - 3)), font, scale, (0, 0, 0), 1, cv2.LINE_AA)

    # 4. Render Footpoints
    for fp in footpoints:
        color = _get_track_color(fp.track_id)
        fx, fy = int(round(fp.x)), int(round(fp.y))
        cv2.circle(output, (fx, fy), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(output, (fx, fy), 6, color, 1, cv2.LINE_AA)

    # 5. Diagnostic HUD Header Banner
    hud_h = 32
    hud_overlay = output.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (15, 15, 15), -1)
    cv2.addWeighted(hud_overlay, 0.75, output, 0.25, 0, output)

    in_zone_count = sum(1 for m in memberships if m.is_in_zone)
    fps_str = f" | Throughput: {throughput_fps:.1f} FPS" if throughput_fps is not None else ""
    hud_text = (
        f"Frame: {frame_index:04d} | Active Tracks: {len(tracks)} | "
        f"Footpoints: {len(footpoints)} | In-Zone: {in_zone_count} | Registered Zones: {len(zones)}"
        f"{fps_str}"
    )
    cv2.putText(output, hud_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return output
