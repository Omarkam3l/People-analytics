"""Scene configuration and calibration presets for Phase 12 full-system evaluation."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from people_analytics.dwell.models import DwellConfig
from people_analytics.ground.homography import create_ground_calibration
from people_analytics.ground.models import (
    CalibrationPoint,
    CoordinateFrame,
    GroundPlaneCalibration,
)
from people_analytics.ground_analytics.models import GroundHeatmapConfig, GroundZone
from people_analytics.heatmap.models import HeatmapConfig
from people_analytics.zone.models import Zone


@dataclass(frozen=True)
class SceneEvaluationConfig:
    """Complete scene setup for full-system dual-space evaluation."""
    sequence_name: str
    image_width: int
    image_height: int
    fps: float
    image_zones: Sequence[Zone]
    ground_calibration: GroundPlaneCalibration
    ground_zones: Sequence[GroundZone]
    image_heatmap_config: HeatmapConfig
    ground_heatmap_config: GroundHeatmapConfig
    dwell_config: DwellConfig


def get_mot17_09_evaluation_config() -> SceneEvaluationConfig:
    """Controlled manual calibration and spatial zones for primary sequence MOT17-09-FRCNN.

    Scene Geometry:
        - Image canvas: 1920x1080 @ 30 FPS.
        - Ground frame: CoordinateFrame.ARBITRARY_PLANAR (arbitrary units, NOT meters).
        - Corridor ground span: X in [0.0, 20.0], Y in [0.0, 8.0].
    """
    width = 1920
    height = 1080
    fps = 30.0

    # Controlled 4-point manual calibration fixture for MOT17-09
    ref_pts = [
        CalibrationPoint(image_x=200.0, image_y=550.0, ground_x=0.0, ground_y=8.0, label="Far_Left"),
        CalibrationPoint(image_x=1500.0, image_y=550.0, ground_x=20.0, ground_y=8.0, label="Far_Right"),
        CalibrationPoint(image_x=1750.0, image_y=950.0, ground_x=20.0, ground_y=0.0, label="Near_Right"),
        CalibrationPoint(image_x=150.0, image_y=950.0, ground_x=0.0, ground_y=0.0, label="Near_Left"),
    ]
    calib = create_ground_calibration(
        calibration_id="calib_mot17_09_planar",
        scene_id="MOT17-09-FRCNN",
        source_image_size=(width, height),
        target_frame=CoordinateFrame.ARBITRARY_PLANAR,
        units="arbitrary",
        reference_points=ref_pts,
        metadata={"description": "Controlled manual planar calibration for MOT17-09 scene corridor"},
    )

    # Image Zones: Central corridor and Left periphery (overlapping)
    img_zone1 = Zone(
        zone_id="Zone_Central",
        name="Central Corridor",
        vertices=(
            (0.20 * width, 0.40 * height),
            (0.80 * width, 0.40 * height),
            (0.80 * width, 0.85 * height),
            (0.20 * width, 0.85 * height),
        ),
    )
    img_zone2 = Zone(
        zone_id="Zone_Left",
        name="Left Entrance",
        vertices=(
            (0.05 * width, 0.35 * height),
            (0.35 * width, 0.35 * height),
            (0.35 * width, 0.90 * height),
            (0.05 * width, 0.90 * height),
        ),
    )

    # Ground Zones: Overlapping ground corridor and entrance for multi-zone audit
    grd_zone1 = GroundZone(
        zone_id="Zone_Ground_Central",
        name="Ground Central Corridor",
        vertices=(
            (3.0, 1.0),
            (18.0, 1.0),
            (18.0, 7.0),
            (3.0, 7.0),
        ),
    )
    grd_zone2 = GroundZone(
        zone_id="Zone_Ground_Left",
        name="Ground Left Entrance",
        vertices=(
            (0.0, 0.0),
            (6.0, 0.0),
            (6.0, 8.0),
            (0.0, 8.0),
        ),
    )

    img_hm_cfg = HeatmapConfig(image_width=width, image_height=height, cell_size=50)
    grd_hm_cfg = GroundHeatmapConfig(min_x=0.0, max_x=20.0, min_y=0.0, max_y=8.0, cell_size=0.5)
    dwell_cfg = DwellConfig(fps=fps, max_gap_frames=0)

    return SceneEvaluationConfig(
        sequence_name="MOT17-09-FRCNN",
        image_width=width,
        image_height=height,
        fps=fps,
        image_zones=[img_zone1, img_zone2],
        ground_calibration=calib,
        ground_zones=[grd_zone1, grd_zone2],
        image_heatmap_config=img_hm_cfg,
        ground_heatmap_config=grd_hm_cfg,
        dwell_config=dwell_cfg,
    )


def get_mot17_02_evaluation_config() -> SceneEvaluationConfig:
    """Controlled manual calibration and spatial zones for secondary sequence MOT17-02-FRCNN.

    Scene Geometry:
        - Image canvas: 1920x1080 @ 30 FPS.
        - Ground frame: CoordinateFrame.ARBITRARY_PLANAR.
        - Indoor floor ground span: X in [0.0, 15.0], Y in [0.0, 15.0].
    """
    width = 1920
    height = 1080
    fps = 30.0

    # Controlled 4-point manual calibration fixture for MOT17-02 indoor concourse
    ref_pts = [
        CalibrationPoint(image_x=400.0, image_y=450.0, ground_x=0.0, ground_y=15.0, label="Far_Left"),
        CalibrationPoint(image_x=1550.0, image_y=450.0, ground_x=15.0, ground_y=15.0, label="Far_Right"),
        CalibrationPoint(image_x=1750.0, image_y=950.0, ground_x=15.0, ground_y=0.0, label="Near_Right"),
        CalibrationPoint(image_x=200.0, image_y=950.0, ground_x=0.0, ground_y=0.0, label="Near_Left"),
    ]
    calib = create_ground_calibration(
        calibration_id="calib_mot17_02_planar",
        scene_id="MOT17-02-FRCNN",
        source_image_size=(width, height),
        target_frame=CoordinateFrame.ARBITRARY_PLANAR,
        units="arbitrary",
        reference_points=ref_pts,
        metadata={"description": "Controlled manual planar calibration for MOT17-02 indoor corridor"},
    )

    # Image Zones: Main concourse and right walkway
    img_zone1 = Zone(
        zone_id="Zone_Concourse",
        name="Main Concourse",
        vertices=(
            (0.20 * width, 0.40 * height),
            (0.80 * width, 0.40 * height),
            (0.80 * width, 0.85 * height),
            (0.20 * width, 0.85 * height),
        ),
    )
    img_zone2 = Zone(
        zone_id="Zone_Right",
        name="Right Walkway",
        vertices=(
            (0.60 * width, 0.35 * height),
            (0.95 * width, 0.35 * height),
            (0.95 * width, 0.90 * height),
            (0.60 * width, 0.90 * height),
        ),
    )

    # Ground Zones: Overlapping concourse and right wing for multi-zone audit
    grd_zone1 = GroundZone(
        zone_id="Zone_Ground_Concourse",
        name="Ground Main Concourse",
        vertices=(
            (2.0, 2.0),
            (13.0, 2.0),
            (13.0, 13.0),
            (2.0, 13.0),
        ),
    )
    grd_zone2 = GroundZone(
        zone_id="Zone_Ground_Right",
        name="Ground Right Walkway",
        vertices=(
            (9.0, 1.0),
            (14.5, 1.0),
            (14.5, 14.0),
            (9.0, 14.0),
        ),
    )

    img_hm_cfg = HeatmapConfig(image_width=width, image_height=height, cell_size=50)
    grd_hm_cfg = GroundHeatmapConfig(min_x=0.0, max_x=15.0, min_y=0.0, max_y=15.0, cell_size=0.5)
    dwell_cfg = DwellConfig(fps=fps, max_gap_frames=0)

    return SceneEvaluationConfig(
        sequence_name="MOT17-02-FRCNN",
        image_width=width,
        image_height=height,
        fps=fps,
        image_zones=[img_zone1, img_zone2],
        ground_calibration=calib,
        ground_zones=[grd_zone1, grd_zone2],
        image_heatmap_config=img_hm_cfg,
        ground_heatmap_config=grd_hm_cfg,
        dwell_config=dwell_cfg,
    )
