"""Unit and integration tests for Phase 10: Ground-Plane Projection & Homography."""

import math
import numpy as np
import pytest

from people_analytics.footpoint.models import FootpointObservation
from people_analytics.ground import (
    CalibrationPoint,
    CoordinateFrame,
    GroundPlaneCalibration,
    GroundPlaneProjector,
    GroundPoint,
    check_collinearity,
    compute_reprojection_errors,
    create_ground_calibration,
    draw_ground_plane_map,
    estimate_homography,
    normalize_points_hartley,
)
from people_analytics.trajectory.models import Trajectory


# ====================================================================
# 1. CalibrationPoint & Model Tests
# ====================================================================

def test_calibration_point_valid():
    """Verify valid CalibrationPoint creation and immutability."""
    pt = CalibrationPoint(image_x=100.0, image_y=200.0, ground_x=1.5, ground_y=3.0, label="P1")
    assert pt.image_x == 100.0
    assert pt.image_y == 200.0
    assert pt.ground_x == 1.5
    assert pt.ground_y == 3.0
    assert pt.label == "P1"

    with pytest.raises(AttributeError):
        pt.image_x = 200.0  # type: ignore


def test_calibration_point_non_finite_raises():
    """Verify non-finite coordinates in CalibrationPoint raise ValueError."""
    with pytest.raises(ValueError, match="must be finite"):
        CalibrationPoint(image_x=float("nan"), image_y=100.0, ground_x=0.0, ground_y=0.0)

    with pytest.raises(ValueError, match="must be finite"):
        CalibrationPoint(image_x=100.0, image_y=float("inf"), ground_x=0.0, ground_y=0.0)


# ====================================================================
# 2. Geometry & Hartley Normalization Tests
# ====================================================================

def test_check_collinearity_passes_on_rectangle():
    """Four corners of a rectangle have no collinear triplets."""
    pts = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    # Should not raise
    check_collinearity(pts)


def test_check_collinearity_raises_on_horizontal_line():
    """Three points on a horizontal line must raise ValueError."""
    pts = np.array([[0.0, 50.0], [50.0, 50.0], [100.0, 50.0], [20.0, 80.0]])
    with pytest.raises(ValueError, match="collinear"):
        check_collinearity(pts)


def test_check_collinearity_raises_on_diagonal_line():
    """Three points on a diagonal line must raise ValueError."""
    pts = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [5.0, 40.0]])
    with pytest.raises(ValueError, match="collinear"):
        check_collinearity(pts)


def test_normalize_points_hartley():
    """Hartley normalization must yield zero mean and average distance sqrt(2)."""
    pts = np.array([
        [100.0, 150.0],
        [400.0, 150.0],
        [450.0, 500.0],
        [50.0, 500.0],
    ])
    T, norm_pts = normalize_points_hartley(pts)

    assert T.shape == (3, 3)
    assert norm_pts.shape == (4, 2)

    # Centroid should be zero
    centroid = np.mean(norm_pts, axis=0)
    assert np.allclose(centroid, [0.0, 0.0], atol=1e-10)

    # Average distance to origin should be sqrt(2)
    mean_dist = np.mean(np.sqrt(np.sum(norm_pts ** 2, axis=1)))
    assert np.isclose(mean_dist, math.sqrt(2.0), atol=1e-10)


def test_normalize_points_hartley_zero_spread_raises():
    """Identical points with zero spread cannot be normalized."""
    pts = np.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0], [10.0, 10.0]])
    with pytest.raises(ValueError, match="spatial spread"):
        normalize_points_hartley(pts)


# ====================================================================
# 3. Homography Estimation Tests
# ====================================================================

def test_estimate_homography_identity():
    """Mapping points to themselves must produce the identity homography."""
    src = [(100.0, 100.0), (500.0, 100.0), (500.0, 400.0), (100.0, 400.0)]
    H = estimate_homography(src, src)

    assert H.shape == (3, 3)
    assert np.allclose(H, np.eye(3), atol=1e-7)


def test_estimate_homography_translation_and_scale():
    """Homography with known pure translation (+50, +30) and scaling (2x)."""
    src = [(10.0, 20.0), (110.0, 20.0), (110.0, 120.0), (10.0, 120.0)]
    scale = 2.0
    tx, ty = 50.0, 30.0
    dst = [(x * scale + tx, y * scale + ty) for x, y in src]

    H = estimate_homography(src, dst)

    expected_H = np.array([
        [scale, 0.0, tx],
        [0.0, scale, ty],
        [0.0, 0.0, 1.0],
    ])
    assert np.allclose(H, expected_H, atol=1e-6)


def test_estimate_homography_perspective_trapezoid():
    """Perspective projection mapping a trapezoid to a rectangle."""
    # Image trapezoid (perspective road/sidewalk)
    src = [(200.0, 100.0), (400.0, 100.0), (500.0, 400.0), (100.0, 400.0)]
    # Metric ground rectangle: 3 meters wide, 10 meters long
    dst = [(0.0, 10.0), (3.0, 10.0), (3.0, 0.0), (0.0, 0.0)]

    H = estimate_homography(src, dst)

    # Project each point through H
    for (sx, sy), (dx, dy) in zip(src, dst):
        p = H @ np.array([sx, sy, 1.0])
        proj_x = p[0] / p[2]
        proj_y = p[1] / p[2]
        assert np.isclose(proj_x, dx, atol=1e-6)
        assert np.isclose(proj_y, dy, atol=1e-6)


def test_estimate_homography_inverse_consistency():
    """H^-1 @ H must equal the 3x3 identity matrix within 1e-6."""
    src = [(150.0, 80.0), (420.0, 110.0), (490.0, 390.0), (90.0, 350.0)]
    dst = [(0.0, 5.0), (2.5, 5.0), (2.5, 0.0), (0.0, 0.0)]

    H = estimate_homography(src, dst)
    H_inv = np.linalg.inv(H)
    identity = H_inv @ H
    identity = identity / identity[2, 2]

    assert np.allclose(identity, np.eye(3), atol=1e-6)


def test_estimate_homography_fewer_than_four_points_raises():
    """Homography with < 4 points must raise ValueError."""
    src = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    dst = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    with pytest.raises(ValueError, match="at least 4 point correspondences"):
        estimate_homography(src, dst)


def test_estimate_homography_collinear_raises():
    """Collinear calibration points must raise ValueError."""
    src = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (5.0, 0.0)]
    dst = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    with pytest.raises(ValueError, match="collinear"):
        estimate_homography(src, dst)


# ====================================================================
# 4. Reprojection & Calibration Factory Tests
# ====================================================================

def test_compute_reprojection_errors_perfect():
    """Exact mapping should yield zero reprojection error."""
    src = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    dst = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
    H = estimate_homography(src, dst)

    errs = compute_reprojection_errors(H, src, dst)
    assert np.isclose(errs["rmse_forward"], 0.0, atol=1e-6)
    assert np.isclose(errs["max_forward"], 0.0, atol=1e-6)
    assert np.isclose(errs["rmse_backward"], 0.0, atol=1e-6)


def test_create_ground_calibration_metric():
    """Verify factory builds valid GroundPlaneCalibration with metric units."""
    ref_pts = [
        CalibrationPoint(100.0, 100.0, 0.0, 5.0, label="TL"),
        CalibrationPoint(300.0, 100.0, 3.0, 5.0, label="TR"),
        CalibrationPoint(350.0, 400.0, 3.0, 0.0, label="BR"),
        CalibrationPoint(50.0, 400.0, 0.0, 0.0, label="BL"),
    ]

    calib = create_ground_calibration(
        calibration_id="calib_mot09",
        scene_id="MOT17-09",
        source_image_size=(1920, 1080),
        target_frame=CoordinateFrame.METRIC_GROUND,
        units="meters",
        reference_points=ref_pts,
        metadata={"camera_type": "static_surveillance"},
    )

    assert calib.calibration_id == "calib_mot09"
    assert calib.scene_id == "MOT17-09"
    assert calib.target_frame == CoordinateFrame.METRIC_GROUND
    assert calib.units == "meters"
    assert len(calib.reference_points) == 4
    assert len(calib.homography_matrix) == 3
    assert calib.reprojection_rmse < 1e-5
    assert calib.metadata["camera_type"] == "static_surveillance"


def test_create_ground_calibration_metric_requires_meters():
    """Specifying METRIC_GROUND with units='pixels' must raise ValueError."""
    ref_pts = [
        CalibrationPoint(0.0, 0.0, 0.0, 10.0),
        CalibrationPoint(100.0, 0.0, 10.0, 10.0),
        CalibrationPoint(100.0, 100.0, 10.0, 0.0),
        CalibrationPoint(0.0, 100.0, 0.0, 0.0),
    ]
    with pytest.raises(ValueError, match="metric frame requires units='meters'"):
        create_ground_calibration(
            calibration_id="bad_calib",
            scene_id="scene1",
            source_image_size=(1920, 1080),
            target_frame=CoordinateFrame.METRIC_GROUND,
            units="pixels",
            reference_points=ref_pts,
        )


# ====================================================================
# 5. GroundPlaneProjector Tests
# ====================================================================

def test_projector_point_projection():
    """Verify point projection, extrapolation check, and back-projection."""
    ref_pts = [
        CalibrationPoint(100.0, 100.0, 0.0, 10.0),
        CalibrationPoint(400.0, 100.0, 4.0, 10.0),
        CalibrationPoint(450.0, 500.0, 4.0, 0.0),
        CalibrationPoint(50.0, 500.0, 0.0, 0.0),
    ]
    calib = create_ground_calibration(
        calibration_id="test_calib",
        scene_id="test_scene",
        source_image_size=(1920, 1080),
        target_frame=CoordinateFrame.METRIC_GROUND,
        units="meters",
        reference_points=ref_pts,
    )
    projector = GroundPlaneProjector(calib)

    # Project inside point: (250, 300) is center of trapezoid
    gp_inside = projector.project_point(250.0, 300.0, track_id=1, frame_index=10)
    assert gp_inside.is_valid is True
    assert gp_inside.is_extrapolated is False
    assert gp_inside.track_id == 1
    assert gp_inside.frame_index == 10
    assert 0.0 <= gp_inside.x <= 4.0
    assert 0.0 <= gp_inside.y <= 10.0

    # Back-project to image
    img_x, img_y = projector.project_ground_to_image(gp_inside.x, gp_inside.y)
    assert np.isclose(img_x, 250.0, atol=1e-5)
    assert np.isclose(img_y, 300.0, atol=1e-5)

    # Project outside point: (10.0, 10.0) is far outside calibration trapezoid
    gp_outside = projector.project_point(10.0, 10.0, track_id=2, frame_index=10)
    assert gp_outside.is_valid is True
    assert gp_outside.is_extrapolated is True


def test_projector_non_finite_input():
    """Non-finite coordinates produce invalid GroundPoint with NON_FINITE_INPUT code."""
    ref_pts = [
        CalibrationPoint(0.0, 0.0, 0.0, 1.0),
        CalibrationPoint(10.0, 0.0, 1.0, 1.0),
        CalibrationPoint(10.0, 10.0, 1.0, 0.0),
        CalibrationPoint(0.0, 10.0, 0.0, 0.0),
    ]
    calib = create_ground_calibration(
        calibration_id="calib",
        scene_id="scene",
        source_image_size=(100, 100),
        target_frame=CoordinateFrame.ARBITRARY_PLANAR,
        units="arbitrary",
        reference_points=ref_pts,
    )
    projector = GroundPlaneProjector(calib)

    gp = projector.project_point(float("nan"), 50.0)
    assert gp.is_valid is False
    assert gp.error_code == "NON_FINITE_INPUT"
    assert math.isnan(gp.x)


def test_projector_horizon_rejection():
    """Points at/above horizon with w' <= 0 are rejected as invalid."""
    # Create perspective mapping where high Y in image goes far away, low Y goes to horizon
    ref_pts = [
        CalibrationPoint(200.0, 200.0, 0.0, 100.0),
        CalibrationPoint(400.0, 200.0, 10.0, 100.0),
        CalibrationPoint(500.0, 500.0, 10.0, 1.0),
        CalibrationPoint(100.0, 500.0, 0.0, 1.0),
    ]
    calib = create_ground_calibration(
        calibration_id="horizon_calib",
        scene_id="scene",
        source_image_size=(1920, 1080),
        target_frame=CoordinateFrame.METRIC_GROUND,
        units="meters",
        reference_points=ref_pts,
    )
    projector = GroundPlaneProjector(calib)

    # Point at extreme top of image (above horizon line where w' <= 0)
    gp_horizon = projector.project_point(300.0, -1000.0)
    assert gp_horizon.is_valid is False
    assert gp_horizon.is_extrapolated is False
    assert gp_horizon.error_code == "PROJECTIVE_DENOMINATOR_TOO_SMALL"
    assert gp_horizon.frame == CoordinateFrame.METRIC_GROUND


def test_manual_calibration_with_held_out_validation_points():
    """Verify Controlled Manual Planar Calibration Demonstration with internal consistency check points."""
    # 6 total correspondences: 4 fitting points + 2 model-derived internal consistency check points
    fitting_pts = [
        CalibrationPoint(image_x=200.0, image_y=550.0, ground_x=0.0, ground_y=8.0, label="Far_Left"),
        CalibrationPoint(image_x=1500.0, image_y=550.0, ground_x=20.0, ground_y=8.0, label="Far_Right"),
        CalibrationPoint(image_x=1750.0, image_y=950.0, ground_x=20.0, ground_y=0.0, label="Near_Right"),
        CalibrationPoint(image_x=150.0, image_y=950.0, ground_x=0.0, ground_y=0.0, label="Near_Left"),
    ]

    # These 2 points were mathematically derived from the corridor geometry and rounded to integer pixels
    internal_consistency_pts = [
        CalibrationPoint(image_x=850.0, image_y=550.0, ground_x=10.0, ground_y=8.0, label="Far_Mid"),
        CalibrationPoint(image_x=895.0, image_y=729.0, ground_x=10.0, ground_y=4.0, label="Center_Corridor"),
    ]

    calib = create_ground_calibration(
        calibration_id="calib_mot17_09_manual_planar",
        scene_id="MOT17-09-FRCNN",
        source_image_size=(1920, 1080),
        target_frame=CoordinateFrame.ARBITRARY_PLANAR,
        units="arbitrary",
        reference_points=fitting_pts,
        metadata={"demonstration_name": "Controlled Manual Planar Calibration Demonstration"},
    )

    # 1. Fit/Reprojection RMSE on 4 fitting points
    assert calib.reprojection_rmse < 1e-10
    assert calib.units == "arbitrary"
    assert calib.target_frame == CoordinateFrame.ARBITRARY_PLANAR

    # 2. Internal consistency check on the 2 held-out points
    projector = GroundPlaneProjector(calib)
    val_errors = []
    for p in internal_consistency_pts:
        gp = projector.project_point(p.image_x, p.image_y)
        assert gp.is_valid is True
        assert gp.is_extrapolated is False
        assert gp.error_code is None
        assert gp.frame == CoordinateFrame.ARBITRARY_PLANAR
        err = math.sqrt((gp.x - p.ground_x) ** 2 + (gp.y - p.ground_y) ** 2)
        val_errors.append(err)

    val_rmse = math.sqrt(sum(e ** 2 for e in val_errors) / len(val_errors))
    # Consistency RMSE reflects pixel discretization residual on model-derived coordinates (~0.005 arbitrary units)
    assert val_rmse < 0.01
    assert val_rmse > 1e-6  # Non-zero due to pixel rounding


def test_projector_extrapolated_but_valid_projection():
    """Points outside the calibration polygon remain is_valid=True with is_extrapolated=True."""
    ref_pts = [
        CalibrationPoint(100.0, 100.0, 0.0, 10.0),
        CalibrationPoint(400.0, 100.0, 4.0, 10.0),
        CalibrationPoint(450.0, 500.0, 4.0, 0.0),
        CalibrationPoint(50.0, 500.0, 0.0, 0.0),
    ]
    calib = create_ground_calibration(
        calibration_id="extrap_calib",
        scene_id="scene",
        source_image_size=(1920, 1080),
        target_frame=CoordinateFrame.ARBITRARY_PLANAR,
        units="arbitrary",
        reference_points=ref_pts,
    )
    projector = GroundPlaneProjector(calib)

    # Point outside convex hull: (10.0, 10.0)
    gp = projector.project_point(10.0, 10.0, track_id=42, frame_index=3)
    assert gp.is_valid is True
    assert gp.is_extrapolated is True
    assert gp.error_code is None
    assert gp.frame == CoordinateFrame.ARBITRARY_PLANAR
    assert math.isfinite(gp.x)
    assert math.isfinite(gp.y)


def test_ground_point_coordinate_frame_semantics():
    """Verify coordinate-frame propagation and explicit distinction."""
    gp_arb = GroundPoint(track_id=1, frame_index=1, x=5.0, y=10.0, is_valid=True, frame=CoordinateFrame.ARBITRARY_PLANAR)
    assert gp_arb.frame == CoordinateFrame.ARBITRARY_PLANAR
    assert gp_arb.frame.value == "arbitrary_planar"

    gp_metric = GroundPoint(track_id=1, frame_index=1, x=5.0, y=10.0, is_valid=True, frame=CoordinateFrame.METRIC_GROUND)
    assert gp_metric.frame == CoordinateFrame.METRIC_GROUND
    assert gp_metric.frame.value == "metric_ground"


def test_real_mot17_09_manual_calibration_fixture():
    """Reproducible calibration fixture for MOT17-09-FRCNN scene corridor."""
    ref_pts = [
        CalibrationPoint(image_x=200.0, image_y=550.0, ground_x=0.0, ground_y=8.0, label="Far_Left"),
        CalibrationPoint(image_x=1500.0, image_y=550.0, ground_x=20.0, ground_y=8.0, label="Far_Right"),
        CalibrationPoint(image_x=1750.0, image_y=950.0, ground_x=20.0, ground_y=0.0, label="Near_Right"),
        CalibrationPoint(image_x=150.0, image_y=950.0, ground_x=0.0, ground_y=0.0, label="Near_Left"),
    ]

    calib = create_ground_calibration(
        calibration_id="calib_mot17_09_manual_planar",
        scene_id="MOT17-09-FRCNN",
        source_image_size=(1920, 1080),
        target_frame=CoordinateFrame.ARBITRARY_PLANAR,
        units="arbitrary",
        reference_points=ref_pts,
        metadata={"description": "Reproducible controlled manual calibration for MOT17-09 scene"},
    )

    # 1. Check calibration quality
    assert calib.reprojection_rmse < 1e-10
    H = np.array(calib.homography_matrix)
    H_inv = np.array(calib.inverse_homography_matrix)
    eye = H @ H_inv
    eye = eye / eye[2, 2]
    assert np.allclose(eye, np.eye(3), atol=1e-5)

    # 2. Project known footpoints from MOT17-09 frame 5
    projector = GroundPlaneProjector(calib)

    # Track 2: (310.0, 707.7) -> inside sidewalk corridor
    gp_trk2 = projector.project_point(310.0, 707.7, track_id=2, frame_index=5)
    assert gp_trk2.is_valid is True
    assert gp_trk2.is_extrapolated is False
    assert gp_trk2.error_code is None
    assert gp_trk2.frame == CoordinateFrame.ARBITRARY_PLANAR
    assert 0.0 <= gp_trk2.x <= 20.0
    assert 0.0 <= gp_trk2.y <= 8.0

    # Track 8: (1898.2, 586.6) -> outside corridor to the right
    gp_trk8 = projector.project_point(1898.2, 586.6, track_id=8, frame_index=5)
    assert gp_trk8.is_valid is True
    assert gp_trk8.is_extrapolated is True
    assert gp_trk8.error_code is None
    assert gp_trk8.x > 20.0  # to the right of corridor


def test_projector_observation_and_trajectory_integration():
    """Verify FootpointObservation and Trajectory batch projection."""
    ref_pts = [
        CalibrationPoint(0.0, 0.0, 0.0, 10.0),
        CalibrationPoint(100.0, 0.0, 10.0, 10.0),
        CalibrationPoint(100.0, 100.0, 10.0, 0.0),
        CalibrationPoint(0.0, 100.0, 0.0, 0.0),
    ]
    calib = create_ground_calibration(
        calibration_id="integ_calib",
        scene_id="scene",
        source_image_size=(100, 100),
        target_frame=CoordinateFrame.METRIC_GROUND,
        units="meters",
        reference_points=ref_pts,
    )
    projector = GroundPlaneProjector(calib)

    fp1 = FootpointObservation(track_id=7, frame_index=1, x=20.0, y=20.0, confidence=0.9)
    fp2 = FootpointObservation(track_id=7, frame_index=2, x=30.0, y=30.0, confidence=0.9)
    fp3 = FootpointObservation(track_id=7, frame_index=5, x=50.0, y=50.0, confidence=0.9)  # gap

    traj = Trajectory(track_id=7, points=(fp1, fp2, fp3))

    ground_points = projector.project_trajectory(traj)
    assert len(ground_points) == 3
    assert all(gp.track_id == 7 for gp in ground_points)
    assert [gp.frame_index for gp in ground_points] == [1, 2, 5]
    assert all(gp.is_valid for gp in ground_points)


# ====================================================================
# 6. Visualization Tests
# ====================================================================

def test_draw_ground_plane_map():
    """Verify bird's-eye visualization renders without error and preserves canvas."""
    ref_pts = [
        CalibrationPoint(100.0, 100.0, 0.0, 10.0, label="P1"),
        CalibrationPoint(400.0, 100.0, 5.0, 10.0, label="P2"),
        CalibrationPoint(450.0, 500.0, 5.0, 0.0, label="P3"),
        CalibrationPoint(50.0, 500.0, 0.0, 0.0, label="P4"),
    ]
    calib = create_ground_calibration(
        calibration_id="vis_calib",
        scene_id="MOT17-09",
        source_image_size=(1920, 1080),
        target_frame=CoordinateFrame.METRIC_GROUND,
        units="meters",
        reference_points=ref_pts,
    )

    pts = [
        GroundPoint(track_id=1, frame_index=1, x=2.5, y=5.0, is_valid=True),
        GroundPoint(track_id=2, frame_index=1, x=6.0, y=8.0, is_valid=True, is_extrapolated=True),
        GroundPoint(track_id=3, frame_index=1, x=float("nan"), y=float("nan"), is_valid=False),
    ]
    trajs = {
        1: [
            GroundPoint(track_id=1, frame_index=1, x=2.0, y=4.0, is_valid=True),
            GroundPoint(track_id=1, frame_index=2, x=2.5, y=5.0, is_valid=True),
        ]
    }

    img = draw_ground_plane_map(
        calibration=calib,
        points=pts,
        trajectories=trajs,
        canvas_size=(600, 600),
    )

    assert img.shape == (600, 600, 3)
    assert img.dtype == np.uint8
    # Canvas should not be empty
    assert np.any(img > 0)
