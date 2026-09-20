"""Comprehensive unit and integration tests for Phase 11: Ground-Plane Analytics."""

import math
import numpy as np
import pytest

from people_analytics.dwell.models import DwellConfig, ZoneVisit
from people_analytics.footpoint.models import FootpointObservation
from people_analytics.ground.models import (
    CalibrationPoint,
    CoordinateFrame,
    GroundPlaneCalibration,
    GroundPoint,
)
from people_analytics.ground.projector import GroundPlaneProjector
from people_analytics.ground_analytics import (
    GroundComparisonReport,
    GroundDwellEngine,
    GroundHeatmapAccumulator,
    GroundHeatmapConfig,
    GroundHeatmapData,
    GroundObservation,
    GroundTrajectory,
    GroundTrajectoryBuilder,
    GroundZone,
    GroundZoneEngine,
    GroundZoneMembership,
    compare_image_and_ground_analytics,
    draw_dual_view_diagnostics,
    draw_ground_analytics_map,
    point_in_ground_polygon,
)
from people_analytics.dwell.engine import DwellTimeEngine
from people_analytics.zone.engine import ZoneEngine
from people_analytics.zone.models import Zone, ZoneMembership


# ====================================================================
# 1. GroundObservation & Provenance Tests
# ====================================================================

def test_ground_observation_valid():
    """Verify valid GroundObservation creation and provenance."""
    obs = GroundObservation(
        track_id=1,
        frame_index=10,
        x=5.25,
        y=12.50,
        frame=CoordinateFrame.ARBITRARY_PLANAR,
        is_valid=True,
        is_extrapolated=False,
        source_footpoint_x=350.0,
        source_footpoint_y=720.0,
    )
    assert obs.track_id == 1
    assert obs.frame_index == 10
    assert obs.x == 5.25
    assert obs.y == 12.50
    assert obs.coordinates == (5.25, 12.50)
    assert obs.source_footpoint_x == 350.0
    assert obs.source_footpoint_y == 720.0
    assert obs.is_valid is True
    assert obs.is_extrapolated is False
    assert obs.error_code is None


def test_ground_observation_from_ground_point_factory():
    """Verify factory preserves full GroundPoint data and source footpoint provenance."""
    gp = GroundPoint(
        track_id=2,
        frame_index=5,
        x=-1.5,
        y=3.0,
        is_valid=True,
        is_extrapolated=True,
        error_code=None,
        frame=CoordinateFrame.ARBITRARY_PLANAR,
    )
    source_fp = FootpointObservation(track_id=2, frame_index=5, x=120.0, y=910.0, confidence=0.88)

    obs = GroundObservation.from_ground_point(gp, source_fp)
    assert obs.track_id == 2
    assert obs.frame_index == 5
    assert obs.x == -1.5
    assert obs.y == 3.0
    assert obs.is_extrapolated is True
    assert obs.source_footpoint_x == 120.0
    assert obs.source_footpoint_y == 910.0


def test_ground_observation_validation_errors():
    """Verify non-finite coordinates on valid observations and negative IDs raise ValueError."""
    with pytest.raises(ValueError, match="track_id"):
        GroundObservation(track_id=-1, frame_index=1, x=0.0, y=0.0)

    with pytest.raises(ValueError, match="frame_index"):
        GroundObservation(track_id=1, frame_index=0, x=0.0, y=0.0)

    with pytest.raises(ValueError, match="finite"):
        GroundObservation(track_id=1, frame_index=1, x=float("nan"), y=0.0, is_valid=True)


# ====================================================================
# 2. GroundTrajectory & Builder Tests
# ====================================================================

def test_ground_trajectory_builder_single_track():
    """Verify builder preserves ordering, gap honesty, and calculates metrics."""
    builder = GroundTrajectoryBuilder()

    obs1 = GroundObservation(1, 1, 0.0, 0.0)
    obs2 = GroundObservation(1, 2, 1.0, 1.0)
    obs3 = GroundObservation(1, 5, 4.0, 4.0)  # gap between 2 and 5

    assert builder.add_observation(obs1) is True
    assert builder.add_observation(obs2) is True
    assert builder.add_observation(obs3) is True

    traj = builder.build(1)
    assert traj.track_id == 1
    assert traj.length == 3
    assert traj.start_frame == 1
    assert traj.end_frame == 5
    assert traj.has_gaps is True
    assert traj.coordinates == [(0.0, 0.0), (1.0, 1.0), (4.0, 4.0)]


def test_ground_trajectory_builder_ignores_invalid():
    """Verify invalid projections are safely ignored from trajectories."""
    builder = GroundTrajectoryBuilder()

    valid_obs = GroundObservation(1, 1, 1.0, 1.0, is_valid=True)
    invalid_obs = GroundObservation(1, 2, float("nan"), float("nan"), is_valid=False, error_code="PROJECTIVE_DENOMINATOR_TOO_SMALL")

    assert builder.add_observation(valid_obs) is True
    assert builder.add_observation(invalid_obs) is False
    assert builder.invalid_ignored_count == 1

    traj = builder.build(1)
    assert traj.length == 1
    assert traj.points[0].frame_index == 1


def test_ground_trajectory_builder_duplicate_frame_raises():
    """Duplicate observations for the same frame_index must raise ValueError."""
    builder = GroundTrajectoryBuilder()
    builder.add_observation(GroundObservation(1, 2, 1.0, 1.0))
    builder.add_observation(GroundObservation(1, 2, 2.0, 2.0))

    with pytest.raises(ValueError, match="Duplicate GroundObservation"):
        builder.build(1)


# ====================================================================
# 3. Ground Heatmap Tests
# ====================================================================

def test_ground_heatmap_config_validation():
    """Verify bounding box and cell_size validation in GroundHeatmapConfig."""
    # Valid
    cfg = GroundHeatmapConfig(min_x=-10.0, max_x=20.0, min_y=-5.0, max_y=15.0, cell_size=2.5)
    assert cfg.grid_width == 12   # (20 - (-10)) / 2.5 = 12
    assert cfg.grid_height == 8   # (15 - (-5)) / 2.5 = 8
    assert cfg.grid_shape == (8, 12)

    with pytest.raises(ValueError, match="greater than min_x"):
        GroundHeatmapConfig(min_x=10.0, max_x=5.0, min_y=0.0, max_y=10.0, cell_size=1.0)

    with pytest.raises(ValueError, match="cell_size"):
        GroundHeatmapConfig(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, cell_size=0.0)


def test_ground_heatmap_accumulation_and_boundary_semantics():
    """Verify accumulation across arbitrary planar coordinates and boundary inclusion."""
    cfg = GroundHeatmapConfig(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, cell_size=2.0)
    acc = GroundHeatmapAccumulator(cfg)

    # Point at (1.0, 1.0) -> col 0, row 0
    assert acc.add_observation(GroundObservation(1, 1, 1.0, 1.0)) is True
    # Point at (9.0, 9.0) -> col 4, row 4
    assert acc.add_observation(GroundObservation(1, 2, 9.0, 9.0)) is True
    # Point exactly on boundary max_x=10.0, max_y=10.0 -> col 4, row 4
    assert acc.add_observation(GroundObservation(1, 3, 10.0, 10.0)) is True
    # Out of bounds point -> rejected without clamping
    assert acc.add_observation(GroundObservation(1, 4, 15.0, 5.0)) is False
    # Invalid point -> rejected
    assert acc.add_observation(GroundObservation(1, 5, float("nan"), 0.0, is_valid=False)) is False

    assert acc.total_accumulated == 3
    assert acc.out_of_bounds_points == 1
    assert acc.invalid_points == 1

    data = acc.build()
    assert data.total_accumulated == 3
    assert data.occupied_cells == 2
    assert data.raw_counts[0, 0] == 1.0
    assert data.raw_counts[4, 4] == 2.0


def test_ground_heatmap_extrapolated_toggle():
    """Verify allow_extrapolated=False rejects extrapolated observations."""
    cfg_strict = GroundHeatmapConfig(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, cell_size=1.0, allow_extrapolated=False)
    acc = GroundHeatmapAccumulator(cfg_strict)

    obs_in = GroundObservation(1, 1, 5.0, 5.0, is_extrapolated=False)
    obs_extrap = GroundObservation(1, 2, 6.0, 6.0, is_extrapolated=True)

    assert acc.add_observation(obs_in) is True
    assert acc.add_observation(obs_extrap) is False
    assert acc.extrapolated_rejected_points == 1
    assert acc.total_accumulated == 1


# ====================================================================
# 4. Ground Zones & Point-in-Polygon Tests
# ====================================================================

def test_point_in_ground_polygon():
    """Verify inclusive ray-casting on ground-plane polygons."""
    poly = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))

    # Interior
    assert point_in_ground_polygon(5.0, 5.0, poly, inclusive=True) is True
    # Exterior
    assert point_in_ground_polygon(12.0, 5.0, poly, inclusive=True) is False
    # Boundary edge
    assert point_in_ground_polygon(5.0, 0.0, poly, inclusive=True) is True
    assert point_in_ground_polygon(5.0, 0.0, poly, inclusive=False) is False
    # Vertex
    assert point_in_ground_polygon(0.0, 0.0, poly, inclusive=True) is True
    assert point_in_ground_polygon(0.0, 0.0, poly, inclusive=False) is False


def test_ground_zone_engine_evaluation():
    """Verify GroundZoneEngine evaluates observations across multiple zones."""
    z1 = GroundZone("Sidewalk", ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)))
    z2 = GroundZone("Entrance", ((5.0, 2.0), (15.0, 2.0), (15.0, 8.0), (5.0, 8.0)))  # Overlaps [5, 10] x [2, 5]

    engine = GroundZoneEngine([z1, z2])
    assert engine.zone_ids == ["Entrance", "Sidewalk"]

    # Point in Sidewalk only
    m1 = engine.evaluate_point(2.0, 2.0)
    assert m1.zone_ids == ("Sidewalk",)
    assert m1.primary_zone_id == "Sidewalk"

    # Point in both overlapping zones: (7.0, 3.0)
    m2 = engine.evaluate_point(7.0, 3.0)
    assert set(m2.zone_ids) == {"Entrance", "Sidewalk"}
    assert m2.primary_zone_id is None  # Multiple matches yields None for primary_zone_id

    # Point outside all zones
    m3 = engine.evaluate_point(20.0, 20.0)
    assert m3.zone_ids == ()
    assert m3.is_in_zone is False


# ====================================================================
# 5. Ground Dwell Engine Adapter Tests
# ====================================================================

def test_ground_dwell_engine_visit_tracking():
    """Verify GroundDwellEngine tracks single and multi-frame visits and handles gaps."""
    config = DwellConfig(fps=30.0, max_gap_frames=0)
    dwell_engine = GroundDwellEngine(config)

    # Track 1: frames 1..5 in 'Zone_A'
    for f in range(1, 6):
        dwell_engine.add_membership(GroundZoneMembership(track_id=1, frame_index=f, x=1.0, y=1.0, zone_ids=("Zone_A",)))

    dwell_engine.finalize_all()
    visits = dwell_engine.visits

    assert len(visits) == 1
    v = visits[0]
    assert v.track_id == 1
    assert v.zone_id == "Zone_A"
    assert v.entry_frame == 1
    assert v.last_observed_frame == 5
    assert v.observation_count == 5
    assert np.isclose(v.duration_seconds, 4.0 / 30.0)


# ====================================================================
# 6. Comparative Analytics Tests
# ====================================================================

def test_compare_image_and_ground_analytics():
    """Verify quantitative comparison report between image and ground analytics."""
    img_m = [
        ZoneMembership(1, 1, 100.0, 200.0, ("Z1",)),
        ZoneMembership(1, 2, 105.0, 205.0, ("Z1",)),
    ]
    grd_m = [
        GroundZoneMembership(1, 1, 2.0, 4.0, ("Z1",)),
        GroundZoneMembership(1, 2, 2.1, 4.1, ("Z1",)),
    ]
    img_v = [ZoneVisit(1, "Z1", 1, 2, 0.0, 0.033, 0.033, 2)]
    grd_v = [ZoneVisit(1, "Z1", 1, 2, 0.0, 0.033, 0.033, 2)]

    report = compare_image_and_ground_analytics(img_m, grd_m, img_v, grd_v)
    assert report.image_visits_total == 1
    assert report.ground_visits_total == 1
    assert report.agreement_ratio == 1.0
    assert report.membership_agreement_count == 2


# ====================================================================
# 7. Visualization Tests
# ====================================================================

def test_draw_ground_analytics_map_and_dual_view():
    """Verify ground map and dual-view composite visualization."""
    z = GroundZone("Corridor", ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)))
    obs = [GroundObservation(1, 1, 5.0, 2.5, is_valid=True)]

    ground_img = draw_ground_analytics_map(zones=[z], active_observations=obs, canvas_size=(400, 400))
    assert ground_img.shape == (400, 400, 3)
    assert ground_img.dtype == np.uint8

    dummy_camera_view = np.zeros((480, 640, 3), dtype=np.uint8)
    dual_img = draw_dual_view_diagnostics(dummy_camera_view, ground_img)
    assert dual_img.shape[0] == 480
    assert dual_img.shape[2] == 3
    assert dual_img.shape[1] > 640  # Combined width


# ====================================================================
# 8. End-to-End Pipeline Integration & Conservation Invariants Test
# ====================================================================

def test_end_to_end_ground_analytics_pipeline():
    """Verify complete flow from FootpointObservation to GroundDwell and audit 4 conservation invariants."""
    # 1. Setup GroundPlaneProjector with synthetic calibration
    calib = GroundPlaneCalibration(
        calibration_id="test_calib",
        scene_id="test_scene",
        source_image_size=(1000, 1000),
        target_frame=CoordinateFrame.ARBITRARY_PLANAR,
        units="arbitrary",
        reference_points=(
            CalibrationPoint(0.0, 0.0, 0.0, 10.0),
            CalibrationPoint(100.0, 0.0, 10.0, 10.0),
            CalibrationPoint(100.0, 100.0, 10.0, 0.0),
            CalibrationPoint(0.0, 100.0, 0.0, 0.0),
        ),
        homography_matrix=((0.1, 0.0, 0.0), (0.0, -0.1, 10.0), (0.0, 0.0, 1.0)),
        inverse_homography_matrix=((10.0, 0.0, 0.0), (0.0, -10.0, 100.0), (0.0, 0.0, 1.0)),
        reprojection_rmse=0.0,
    )
    projector = GroundPlaneProjector(calib)

    # 2. Synthesize footpoints for 2 targets across 3 frames:
    # Target 1: inside calibration ROI
    # Target 2: outside ROI (extrapolated)
    footpoints = [
        FootpointObservation(1, 1, 50.0, 50.0, 0.9),  # Inside
        FootpointObservation(2, 1, 200.0, 50.0, 0.9), # Extrapolated
        FootpointObservation(1, 2, 55.0, 50.0, 0.9),
        FootpointObservation(2, 2, 205.0, 50.0, 0.9),
    ]

    # 3. Project to GroundObservation
    ground_observations = []
    for fp in footpoints:
        gp = projector.project_observation(fp)
        obs = GroundObservation.from_ground_point(gp, source_footpoint=fp)
        ground_observations.append(obs)

    # INVARIANT 1: Projection Conservation
    in_roi_count = sum(1 for o in ground_observations if o.is_valid and not o.is_extrapolated)
    extrap_count = sum(1 for o in ground_observations if o.is_valid and o.is_extrapolated)
    invalid_count = sum(1 for o in ground_observations if not o.is_valid)
    assert len(footpoints) == in_roi_count + extrap_count + invalid_count
    assert in_roi_count == 2
    assert extrap_count == 2
    assert invalid_count == 0

    # 4. Build Trajectories
    traj_builder = GroundTrajectoryBuilder()
    traj_builder.add_observations(ground_observations)
    trajectories = traj_builder.build_all()
    assert len(trajectories) == 2
    assert trajectories[1].length == 2
    assert trajectories[2].length == 2

    # 5. Accumulate Ground Heatmap
    hm_config = GroundHeatmapConfig(min_x=0.0, max_x=30.0, min_y=0.0, max_y=20.0, cell_size=2.0)
    heatmap = GroundHeatmapAccumulator(hm_config)
    heatmap.add_observations(ground_observations)

    # INVARIANT 2: Heatmap Conservation
    assert len(ground_observations) == (
        heatmap.total_accumulated + heatmap.out_of_bounds_points + heatmap.invalid_points + heatmap.extrapolated_rejected_points
    )
    assert heatmap.total_accumulated == 4

    # 6. Evaluate Ground Zones
    zone = GroundZone("Corridor", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
    zone_engine = GroundZoneEngine([zone])
    memberships = [zone_engine.evaluate_observation(o) for o in ground_observations]

    # INVARIANT 3: Ground-Zone Conservation
    inside_count = sum(1 for m in memberships if m.is_in_zone)
    outside_count = sum(1 for m in memberships if not m.is_in_zone)
    assert len(ground_observations) == inside_count + outside_count
    # Target 1 (x=5.0, y=5.0) is inside Corridor; Target 2 (x=20.0, y=5.0) is outside
    assert inside_count == 2
    assert outside_count == 2

    # 7. Ground Dwell Engine
    dwell_engine = GroundDwellEngine(DwellConfig(fps=30.0, max_gap_frames=0))
    dwell_engine.add_memberships(memberships)
    dwell_engine.finalize_all()
    visits = dwell_engine.visits

    # INVARIANT 4: Dwell Zone-Membership Assignment Conservation
    total_dwell_obs = sum(v.observation_count for v in visits)
    zone_membership_assignments = sum(len(m.zone_ids) for m in memberships)
    assert total_dwell_obs <= zone_membership_assignments
    assert total_dwell_obs == zone_membership_assignments  # Exact equality for fully finalized input
    assert len(visits) == 1
    assert visits[0].track_id == 1
    assert visits[0].observation_count == 2


# ====================================================================
# 8. Targeted Hardening & Contract Parity Tests
# ====================================================================

def test_invalid_ground_point_provenance_and_diagnostics():
    """Verify invalid projections are excluded from analytics but remain fully diagnosable."""
    builder = GroundTrajectoryBuilder()
    hm_config = GroundHeatmapConfig(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, cell_size=1.0)
    acc = GroundHeatmapAccumulator(hm_config)

    obs_valid = GroundObservation(1, 1, 5.0, 5.0, is_valid=True)
    obs_denom_err = GroundObservation(
        1, 2, float("nan"), float("nan"), is_valid=False, error_code="PROJECTIVE_DENOMINATOR_TOO_SMALL"
    )
    obs_singular_err = GroundObservation(
        1, 3, float("inf"), float("inf"), is_valid=False, error_code="PROJECTION_MATRIX_SINGULAR"
    )

    # Add to trajectory builder
    assert builder.add_observation(obs_valid) is True
    assert builder.add_observation(obs_denom_err) is False
    assert builder.add_observation(obs_singular_err) is False

    assert builder.invalid_count == 2
    assert builder.invalid_ignored_count == 2
    assert builder.invalid_error_codes == {
        "PROJECTIVE_DENOMINATOR_TOO_SMALL": 1,
        "PROJECTION_MATRIX_SINGULAR": 1,
    }
    assert len(builder.invalid_observations) == 2
    assert builder.invalid_observations[0].error_code == "PROJECTIVE_DENOMINATOR_TOO_SMALL"
    assert builder.invalid_observations[1].error_code == "PROJECTION_MATRIX_SINGULAR"

    # Trajectory points must contain ONLY valid points
    traj = builder.build(1)
    assert traj.length == 1
    assert all(p.is_valid for p in traj.points)

    # Add to heatmap
    assert acc.add_observation(obs_valid) is True
    assert acc.add_observation(obs_denom_err) is False
    assert acc.add_observation(obs_singular_err) is False

    assert acc.invalid_points == 2
    assert acc.invalid_error_codes == {
        "PROJECTIVE_DENOMINATOR_TOO_SMALL": 1,
        "PROJECTION_MATRIX_SINGULAR": 1,
    }
    assert acc.total_accumulated == 1


def test_trajectory_gap_provenance_difference_no_detection_vs_invalid_projection():
    """Verify diagnostic distinction between missing detection (Case A) and invalid projection (Case B)."""
    builder = GroundTrajectoryBuilder()

    # Track 1 across 5 frames:
    # Frame 1: Valid detection & valid projection
    # Frame 2: Missed detection (Case A: no source observation)
    # Frame 3: Valid detection & valid projection
    # Frame 4: Valid detection, but invalid projection (Case B)
    # Frame 5: Valid detection & valid projection
    obs_f1 = GroundObservation(1, 1, 1.0, 1.0, is_valid=True)
    # Frame 2 skipped entirely (Case A)
    obs_f3 = GroundObservation(1, 3, 3.0, 3.0, is_valid=True)
    obs_f4_invalid = GroundObservation(
        1, 4, float("nan"), float("nan"), is_valid=False, error_code="PROJECTIVE_DENOMINATOR_TOO_SMALL"
    )
    obs_f5 = GroundObservation(1, 5, 5.0, 5.0, is_valid=True)

    builder.add_observation(obs_f1)
    builder.add_observation(obs_f3)
    builder.add_observation(obs_f4_invalid)
    builder.add_observation(obs_f5)

    traj = builder.build(1)
    # Spatial trajectory contains only frames 1, 3, 5
    assert traj.length == 3
    assert [p.frame_index for p in traj.points] == [1, 3, 5]
    assert traj.has_gaps is True

    # Diagnostic inspection:
    # Gap between 1 and 3: no invalid observation recorded -> Case A (source observation missed)
    # Gap between 3 and 5: exactly one invalid observation recorded at frame 4 -> Case B (projection failed)
    assert len(builder.invalid_observations) == 1
    invalid_pt = builder.invalid_observations[0]
    assert invalid_pt.frame_index == 4
    assert invalid_pt.error_code == "PROJECTIVE_DENOMINATOR_TOO_SMALL"


def test_extrapolated_point_policy_consistency_and_configuration():
    """Verify default inclusion policy and configurable exclusion across builder, heatmap, zone, and dwell."""
    # Observations: 1 in-bounds, 1 extrapolated
    obs_in = GroundObservation(1, 1, 5.0, 5.0, is_extrapolated=False)
    obs_ext = GroundObservation(1, 2, 8.0, 8.0, is_extrapolated=True)

    # 1. Trajectory Builder
    # Default: included
    builder_default = GroundTrajectoryBuilder(include_extrapolated=True)
    assert builder_default.include_extrapolated is True
    assert builder_default.add_observation(obs_in) is True
    assert builder_default.add_observation(obs_ext) is True
    assert builder_default.build(1).length == 2

    # Configurable: excluded
    builder_strict = GroundTrajectoryBuilder(include_extrapolated=False)
    assert builder_strict.include_extrapolated is False
    assert builder_strict.add_observation(obs_in) is True
    assert builder_strict.add_observation(obs_ext) is False
    assert builder_strict.extrapolated_excluded_count == 1
    assert builder_strict.build(1).length == 1

    # 2. Heatmap
    hm_config_default = GroundHeatmapConfig(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, cell_size=1.0, allow_extrapolated=True)
    acc_default = GroundHeatmapAccumulator(hm_config_default)
    assert acc_default.add_observation(obs_in) is True
    assert acc_default.add_observation(obs_ext) is True
    assert acc_default.total_accumulated == 2

    hm_config_strict = GroundHeatmapConfig(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, cell_size=1.0, allow_extrapolated=False)
    acc_strict = GroundHeatmapAccumulator(hm_config_strict)
    assert acc_strict.add_observation(obs_in) is True
    assert acc_strict.add_observation(obs_ext) is False
    assert acc_strict.extrapolated_rejected_points == 1
    assert acc_strict.total_accumulated == 1

    # 3. Zone Engine
    zone = GroundZone("Zone1", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
    ze_default = GroundZoneEngine([zone], allow_extrapolated=True)
    assert ze_default.allow_extrapolated is True
    assert ze_default.evaluate_observation(obs_ext).zone_ids == ("Zone1",)

    ze_strict = GroundZoneEngine([zone], allow_extrapolated=False)
    assert ze_strict.allow_extrapolated is False
    assert ze_strict.evaluate_observation(obs_ext).zone_ids == ()

    # 4. Dwell Engine with filtered memberships
    dwell_engine = GroundDwellEngine(DwellConfig(fps=10.0, max_gap_frames=0))
    dwell_engine.add_membership(ze_strict.evaluate_observation(obs_ext))
    dwell_engine.finalize_all()
    # Excluded point yields no visit
    assert len(dwell_engine.visits) == 0


def test_ground_heatmap_bounds_negative_coords_and_normalization():
    """Verify arbitrary planar bounds including negative coordinates and normalization."""
    cfg = GroundHeatmapConfig(min_x=-20.0, max_x=20.0, min_y=-10.0, max_y=10.0, cell_size=2.0)
    acc = GroundHeatmapAccumulator(cfg)

    assert cfg.grid_width == 20   # (20 - (-20)) / 2 = 20
    assert cfg.grid_height == 10  # (10 - (-10)) / 2 = 10
    assert cfg.grid_shape == (10, 20)

    # Negative coordinate accumulation
    assert acc.add_observation(GroundObservation(1, 1, -19.0, -9.0)) is True  # col 0, row 0
    assert acc.add_observation(GroundObservation(1, 2, 0.0, 0.0)) is True      # col 10, row 5
    assert acc.add_observation(GroundObservation(1, 3, 20.0, 10.0)) is True   # max boundary -> col 19, row 9

    # Out of bounds rejection
    assert acc.add_observation(GroundObservation(1, 4, -25.0, 0.0)) is False
    assert acc.add_observation(GroundObservation(1, 5, 0.0, 15.0)) is False
    assert acc.out_of_bounds_points == 2
    assert acc.total_accumulated == 3

    # Normalization
    data = acc.build()
    norm_grid = data.to_normalized()
    assert np.isclose(np.max(norm_grid), 1.0)
    assert np.all(norm_grid >= 0.0) and np.all(norm_grid <= 1.0)

    # Accumulator to_normalized()
    acc_norm = acc.to_normalized()
    assert np.array_equal(acc_norm, norm_grid)


def test_ground_zone_semantic_parity_contract():
    """Verify GroundZone and GroundZoneEngine match Phase 7 semantics exactly."""
    # 1. Invalid polygon rejection contract
    with pytest.raises(ValueError, match="at least 3 vertices"):
        GroundZone("Z_TooFew", ((0.0, 0.0), (1.0, 1.0)))

    with pytest.raises(ValueError, match="finite"):
        GroundZone("Z_Inf", ((0.0, 0.0), (1.0, 0.0), (float("inf"), 1.0)))

    with pytest.raises(ValueError, match="(?i)consecutive duplicate"):
        GroundZone("Z_ConsecDup", ((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

    with pytest.raises(ValueError, match="(?i)degenerate"):
        GroundZone("Z_Collinear", ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)))

    with pytest.raises(ValueError, match="intersect"):
        # Bow-tie self-intersection
        GroundZone("Z_BowTie", ((0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0)))

    # 2. Geometric parity comparison with Phase 7 Zone
    poly_vertices = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    p7_zone = Zone("Z_Parity", poly_vertices)
    p11_zone = GroundZone("Z_Parity", poly_vertices)

    test_points = [
        (5.0, 5.0),    # interior
        (5.0, 0.0),    # boundary edge
        (0.0, 0.0),    # vertex
        (10.0, 10.0),  # vertex
        (15.0, 5.0),   # exterior
        (-1.0, 2.0),   # exterior
    ]

    for px, py in test_points:
        for inc in (True, False):
            p7_result = p7_zone.contains_point(px, py, inclusive=inc)
            p11_result = point_in_ground_polygon(px, py, p11_zone.vertices, inclusive=inc)
            assert p11_result == p7_result, f"Parity mismatch at ({px}, {py}) with inclusive={inc}"

    # 3. Deterministic sorted ordering and overlap handling
    z_b = GroundZone("Zone_B", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
    z_a = GroundZone("Zone_A", ((5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)))
    engine = GroundZoneEngine([z_b, z_a])  # added in reverse alphabetical order

    assert engine.zone_ids == ["Zone_A", "Zone_B"]
    matched = engine.evaluate_point_ids(7.0, 7.0)
    assert matched == ("Zone_A", "Zone_B")  # Sorted lexicographically


def test_ground_dwell_semantic_parity_contract():
    """Verify GroundDwellEngine produces exact identical visits to Phase 8 DwellTimeEngine."""
    p8_engine = DwellTimeEngine(DwellConfig(fps=10.0, max_gap_frames=2))
    p11_engine = GroundDwellEngine(DwellConfig(fps=10.0, max_gap_frames=2))

    # Complex sequence:
    # Track 1: single-frame visit in Zone_A at frame 1
    # Track 1: leaves Zone_A at frame 2
    # Track 1: re-enters Zone_A at frame 5, 6, 8 (gap of 2 - within max_gap_frames)
    # Track 2: simultaneous overlap in Zone_A and Zone_B at frames 1..4
    events = [
        # (track_id, frame_index, zone_ids)
        (1, 1, ("Zone_A",)),
        (1, 2, ()),
        (1, 5, ("Zone_A",)),
        (1, 6, ("Zone_A",)),
        (1, 8, ("Zone_A",)),
        (2, 1, ("Zone_A", "Zone_B")),
        (2, 2, ("Zone_A", "Zone_B")),
        (2, 3, ("Zone_A", "Zone_B")),
        (2, 4, ("Zone_A", "Zone_B")),
    ]

    for tid, f, zids in events:
        p8_engine.add_membership(ZoneMembership(track_id=tid, frame_index=f, x=1.0, y=1.0, zone_ids=zids))
        p11_engine.add_membership(GroundZoneMembership(track_id=tid, frame_index=f, x=1.0, y=1.0, zone_ids=zids))

    p8_engine.finalize_all()
    p11_engine.finalize_all()

    v8 = p8_engine.visits
    v11 = p11_engine.visits

    assert len(v11) == len(v8)
    for i in range(len(v8)):
        assert v11[i].track_id == v8[i].track_id
        assert v11[i].zone_id == v8[i].zone_id
        assert v11[i].entry_frame == v8[i].entry_frame
        assert v11[i].last_observed_frame == v8[i].last_observed_frame
        assert v11[i].observation_count == v8[i].observation_count
        assert np.isclose(v11[i].entry_timestamp, v8[i].entry_timestamp)
        assert np.isclose(v11[i].exit_timestamp, v8[i].exit_timestamp)
        assert np.isclose(v11[i].duration_seconds, v8[i].duration_seconds)


def test_comprehensive_conservation_equations():
    """Verify all 4 spatial conservation equations mathematically."""
    # 1. Setup projector and synthetic points
    calib = GroundPlaneCalibration(
        calibration_id="calib_conservation",
        scene_id="scene_conservation",
        source_image_size=(1000, 1000),
        target_frame=CoordinateFrame.ARBITRARY_PLANAR,
        units="arbitrary",
        reference_points=(
            CalibrationPoint(0.0, 0.0, 0.0, 0.0),
            CalibrationPoint(10.0, 0.0, 10.0, 0.0),
            CalibrationPoint(10.0, 10.0, 10.0, 10.0),
            CalibrationPoint(0.0, 10.0, 0.0, 10.0),
        ),
        homography_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        inverse_homography_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        reprojection_rmse=0.0,
    )
    projector = GroundPlaneProjector(calib)

    # 10 footpoint observations
    fps = [
        FootpointObservation(1, 1, 5.0, 5.0),    # in-bounds ROI
        FootpointObservation(1, 2, 7.0, 7.0),    # in-bounds ROI
        FootpointObservation(1, 3, 15.0, 15.0),  # extrapolated (outside ROI)
        FootpointObservation(1, 4, 16.0, 16.0),  # extrapolated (outside ROI)
        FootpointObservation(1, 5, 25.0, 25.0),  # extrapolated (outside ROI)
        FootpointObservation(2, 1, 2.0, 2.0),    # in-bounds ROI
        FootpointObservation(2, 2, 3.0, 3.0),    # in-bounds ROI
        FootpointObservation(2, 3, 30.0, 30.0),  # extrapolated (outside ROI)
        FootpointObservation(2, 4, 32.0, 32.0),  # extrapolated (outside ROI)
        FootpointObservation(2, 5, 35.0, 35.0),  # extrapolated (outside ROI)
    ]
    # Inject 2 invalid projections manually
    invalid_gp1 = GroundPoint(track_id=1, frame_index=6, x=float("nan"), y=float("nan"), is_valid=False, error_code="PROJECTIVE_DENOMINATOR_TOO_SMALL")
    invalid_gp2 = GroundPoint(track_id=2, frame_index=6, x=float("nan"), y=float("nan"), is_valid=False, error_code="PROJECTIVE_DENOMINATOR_TOO_SMALL")

    ground_points = [projector.project_observation(fp) for fp in fps] + [invalid_gp1, invalid_gp2]
    ground_obs = [GroundObservation.from_ground_point(gp) for gp in ground_points]

    # 1. PROJECTION-STATE CONSERVATION (Projector output state)
    # N_projected = N_valid_in_calibration_roi + N_valid_extrapolated + N_invalid
    n_projected = len(ground_obs)
    n_valid_in_roi = sum(1 for o in ground_obs if o.is_valid and not o.is_extrapolated)
    n_valid_extrapolated = sum(1 for o in ground_obs if o.is_valid and o.is_extrapolated)
    n_invalid = sum(1 for o in ground_obs if not o.is_valid)

    assert n_projected == n_valid_in_roi + n_valid_extrapolated + n_invalid
    assert n_projected == 12
    assert n_valid_in_roi == 4       # 4 points inside calibration ROI
    assert n_valid_extrapolated == 6 # 6 points outside calibration ROI
    assert n_invalid == 2            # 2 points with denominator error

    # 2. DOWNSTREAM ANALYTICS ACCEPTANCE (Consumption policy)
    # N_analytics_input = N_valid_in_roi + (N_valid_extrapolated if policy_allows else 0)

    # Case A: Default Policy (include_extrapolated=True / allow_extrapolated=True)
    builder_default = GroundTrajectoryBuilder(include_extrapolated=True)
    for obs in ground_obs:
        builder_default.add_observation(obs)
    n_analytics_default = sum(t.length for t in builder_default.build_all().values())
    assert n_analytics_default == n_valid_in_roi + n_valid_extrapolated  # 4 + 6 = 10
    assert builder_default.invalid_count == n_invalid
    assert builder_default.extrapolated_excluded_count == 0

    # Case B: Strict Policy (include_extrapolated=False / allow_extrapolated=False)
    builder_strict = GroundTrajectoryBuilder(include_extrapolated=False)
    for obs in ground_obs:
        builder_strict.add_observation(obs)
    n_analytics_strict = sum(t.length for t in builder_strict.build_all().values())
    assert n_analytics_strict == n_valid_in_roi  # 4
    assert builder_strict.invalid_count == n_invalid
    assert builder_strict.extrapolated_excluded_count == n_valid_extrapolated  # 6

    # 3. HEATMAP CONSERVATION
    # Heatmap input = accepted + rejected
    hm_config = GroundHeatmapConfig(min_x=0.0, max_x=20.0, min_y=0.0, max_y=20.0, cell_size=2.0, allow_extrapolated=True)
    hm_acc = GroundHeatmapAccumulator(hm_config)
    for obs in ground_obs:
        hm_acc.add_observation(obs)

    n_hm_input = len(ground_obs)
    n_hm_accepted = hm_acc.total_accumulated
    n_hm_rejected = hm_acc.out_of_bounds_points + hm_acc.invalid_points + hm_acc.extrapolated_rejected_points
    assert n_hm_input == n_hm_accepted + n_hm_rejected

    # 4. GROUND-ZONE CONSERVATION
    # Ground zone observations = inside at least one zone + outside all zones
    zone = GroundZone("Zone_ROI", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
    ze = GroundZoneEngine([zone])
    memberships = [ze.evaluate_observation(obs) for obs in ground_obs]

    n_inside = sum(1 for m in memberships if m.is_in_zone)
    n_outside = sum(1 for m in memberships if not m.is_in_zone)
    assert len(ground_obs) == n_inside + n_outside

    # 5. DWELL ZONE-MEMBERSHIP CONSERVATION
    # Total dwell observation assignments <= Total zone-membership assignments
    # (== for fully finalized input)
    dwell_engine = GroundDwellEngine(DwellConfig(fps=10.0, max_gap_frames=2))
    dwell_engine.add_memberships(memberships)
    dwell_engine.finalize_all()

    total_dwell_assignments = sum(v.observation_count for v in dwell_engine.visits)
    total_zone_assignments = sum(len(m.zone_ids) for m in memberships)
    assert total_dwell_assignments <= total_zone_assignments
    assert total_dwell_assignments == total_zone_assignments


def test_dwell_conservation_with_overlapping_zones():
    """Verify Dwell conservation holds strictly when ground observations belong to multiple overlapping zones."""
    # 2 overlapping zones: Zone_A and Zone_B
    zA = GroundZone("Zone_A", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
    zB = GroundZone("Zone_B", ((5.0, 0.0), (15.0, 0.0), (15.0, 10.0), (5.0, 10.0)))
    zone_engine = GroundZoneEngine([zA, zB])

    # 3 observations in the intersection [5, 10] x [0, 10]
    observations = [
        GroundObservation(track_id=1, frame_index=1, x=7.0, y=5.0),
        GroundObservation(track_id=1, frame_index=2, x=7.0, y=5.0),
        GroundObservation(track_id=1, frame_index=3, x=7.0, y=5.0),
    ]

    memberships = [zone_engine.evaluate_observation(obs) for obs in observations]
    # Each observation matches both zones: ("Zone_A", "Zone_B")
    for m in memberships:
        assert m.zone_ids == ("Zone_A", "Zone_B")
        assert len(m.zone_ids) == 2

    # 3 ground observations -> 3 * 2 = 6 zone-membership assignments
    num_ground_obs = len(observations)  # 3
    zone_membership_assignments = sum(len(m.zone_ids) for m in memberships)  # 6
    assert zone_membership_assignments == 6
    assert zone_membership_assignments > num_ground_obs

    # Feed into GroundDwellEngine
    dwell_engine = GroundDwellEngine(DwellConfig(fps=10.0, max_gap_frames=0))
    dwell_engine.add_memberships(memberships)
    dwell_engine.finalize_all()

    visits = dwell_engine.visits
    # Exactly 2 visits: 1 in Zone_A (3 frames) and 1 in Zone_B (3 frames)
    assert len(visits) == 2
    total_dwell_assignments = sum(v.observation_count for v in visits)
    assert total_dwell_assignments == 6

    # Dwell conservation invariant:
    # Total dwell observation assignments == Total zone-membership assignments
    # (and strictly exceeds the count of raw ground observations due to overlap)
    assert total_dwell_assignments == zone_membership_assignments
    assert total_dwell_assignments > num_ground_obs


def test_extrapolated_point_pure_constructor_defaults():
    """Verify that pure zero-argument defaults consistently INCLUDE extrapolated points across all modules."""
    # 1. Trajectory Builder default
    builder = GroundTrajectoryBuilder()
    assert builder.allow_extrapolated is True
    assert builder.include_extrapolated is True

    # 2. Heatmap Config default
    hm_config = GroundHeatmapConfig(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, cell_size=1.0)
    assert hm_config.allow_extrapolated is True
    acc = GroundHeatmapAccumulator(hm_config)

    # 3. Zone Engine default
    zone = GroundZone("Zone_Default", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
    ze = GroundZoneEngine([zone])
    assert ze.allow_extrapolated is True
    assert ze.include_extrapolated is True

    # Test observation with is_extrapolated=True
    obs_ext = GroundObservation(track_id=1, frame_index=1, x=5.0, y=5.0, is_extrapolated=True)

    # Builder accepts by default
    assert builder.add_observation(obs_ext) is True
    assert builder.build(1).length == 1

    # Heatmap accepts by default
    assert acc.add_observation(obs_ext) is True
    assert acc.total_accumulated == 1

    # ZoneEngine matches by default
    m = ze.evaluate_observation(obs_ext)
    assert m.zone_ids == ("Zone_Default",)

    # DwellEngine processes by default
    dwell = GroundDwellEngine(DwellConfig(fps=10.0, max_gap_frames=0))
    dwell.add_membership(m)
    dwell.finalize_all()
    assert len(dwell.visits) == 1
    assert dwell.visits[0].observation_count == 1
