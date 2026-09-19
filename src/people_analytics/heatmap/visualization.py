"""Visualization utilities for image-space heatmaps."""

import math
from typing import Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from people_analytics.heatmap.models import HeatmapData


def render_heatmap(
    data: HeatmapData,
    colormap: int = 2,  # cv2.COLORMAP_JET default value is 2
    sigma: float = 0.0,
    output_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Render a normalized heatmap grid into an RGB/BGR color image.

    Args:
        data: HeatmapData snapshot.
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).
        sigma: Standard deviation for optional Gaussian blur smoothing (0.0 = no smoothing).
        output_shape: Target canvas shape as (height, width). If None, defaults to
            (config.image_height, config.image_width).

    Returns:
        Colorized heatmap as a uint8 BGR numpy array of shape (H, W, 3).
    """
    target_h = output_shape[0] if output_shape is not None else data.config.image_height
    target_w = output_shape[1] if output_shape is not None else data.config.image_width

    grid_norm = data.normalized

    if cv2 is None:
        # Fallback if cv2 is unavailable: return 3-channel grayscale
        res = (grid_norm * 255.0).astype(np.uint8)
        return np.repeat(res[:, :, np.newaxis], 3, axis=2)

    # Upsample if grid dimensions differ from output shape
    if grid_norm.shape != (target_h, target_w):
        grid_resized = cv2.resize(
            grid_norm,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        grid_resized = grid_norm.copy()

    # Optional Gaussian smoothing for presentation
    if sigma > 0.0:
        ksize = int(2 * math.ceil(3 * sigma) + 1)
        grid_smoothed = cv2.GaussianBlur(grid_resized, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        # Re-normalize after smoothing if non-zero
        max_val = float(np.max(grid_smoothed))
        if max_val > 0.0:
            grid_resized = grid_smoothed / max_val
        else:
            grid_resized = grid_smoothed

    # Scale to 8-bit unsigned integer
    uint8_map = (np.clip(grid_resized, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Apply color map
    colorized = cv2.applyColorMap(uint8_map, colormap)
    return colorized


def overlay_heatmap(
    frame: np.ndarray,
    heatmap_bgr: np.ndarray,
    alpha: float = 0.5,
    threshold: float = 0.02,
    density_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Blend a colorized heatmap onto a background image frame.

    Only pixels where spatial density exceeds `threshold` receive the heatmap overlay;
    unvisited regions remain identical to the original background frame.

    Args:
        frame: Source BGR image frame (H, W, 3).
        heatmap_bgr: Colorized heatmap BGR image (H, W, 3).
        alpha: Blending weight for the heatmap in [0.0, 1.0] (1.0 = opaque heatmap).
        threshold: Minimum normalized density threshold in [0.0, 1.0] to display overlay.
        density_map: Optional 2D float normalized density map (H, W) in [0.0, 1.0].
            If None, derived from grayscale intensity of heatmap_bgr.

    Returns:
        New annotated image frame (does not mutate input).
    """
    output = frame.copy()
    if cv2 is None or heatmap_bgr.size == 0:
        return output

    # Ensure heatmap matches frame dimensions
    if heatmap_bgr.shape[:2] != frame.shape[:2]:
        heatmap_bgr = cv2.resize(heatmap_bgr, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Determine mask of active density
    if density_map is not None:
        if density_map.shape != frame.shape[:2]:
            density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = density_map > threshold
    else:
        # Fallback: estimate from heatmap intensity
        gray = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2GRAY)
        mask = (gray / 255.0) > threshold

    if not np.any(mask):
        return output

    # Alpha blend active pixels
    blended = cv2.addWeighted(heatmap_bgr, alpha, frame, 1.0 - alpha, 0.0)
    output[mask] = blended[mask]

    return output
