"""Unit and integration tests for spatial zones layer."""

import numpy as np
import pytest

from people_analytics.footpoint.models import FootpointObservation
from people_analytics.trajectory.models import Trajectory
from people_analytics.zone import (
    Zone,
    ZoneEngine,
    ZoneMembership,
    compute_polygon_area,
    draw_zones,
    is_point_in_polygon,
    is_point_on_segment,
    validate_polygon_vertices,
)


# ====================================================================
# Polygon Geometry & Validation Tests
# ====================================================================

def test_polygon_minimum_vertices():
    with pytest.raises(ValueError, match="at least 3 vertices"):
        Zone(zone_id="too_few", vertices=((0.0, 0.0), (10.0, 10.0)))


def test_polygon_non_finite_coordinates():
    with pytest.raises(ValueError, match="non-finite coordinate"):
        Zone(zone_id="nan_zone", vertices=((0.0, 0.0), (float("nan"), 10.0), (10.0, 0.0)))

    with pytest.raises(ValueError, match="non-finite coordinate"):
        Zone(zone_id="inf_zone", vertices=((0.0, 0.0), (float("inf"), 10.0), (10.0, 0.0)))


def test_polygon_strips_explicit_closing_duplicate():
    # 5 vertices where 5th is identical to 1st
    verts = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0))
    zone = Zone(zone_id="rect", vertices=verts)
    assert len(zone.vertices) == 4
    assert zone.vertices == ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))


def test_polygon_consecutive_duplicates_raise_value_error():
    verts = ((0.0, 0.0), (100.0, 0.0), (100.0, 0.0), (0.0, 100.0))
    with pytest.raises(ValueError, match="consecutive duplicate vertex"):
        Zone(zone_id="dup", vertices=verts)


def test_polygon_degenerate_area_raises_value_error():
    # 3 collinear points
    verts = ((0.0, 0.0), (50.0, 50.0), (100.0, 100.0))
    with pytest.raises(ValueError, match="degenerate or zero area"):
        Zone(zone_id="collinear", vertices=verts)


def test_polygon_self_intersecting_bow_tie_raises_value_error():
    # Bow-tie / figure-8 self-intersecting polygon
    verts = ((0.0, 0.0), (100.0, 100.0), (0.0, 100.0), (100.0, 0.0))
    with pytest.raises(ValueError, match="self-intersecting"):
        Zone(zone_id="bowtie", vertices=verts)


def test_zone_id_validation():
    with pytest.raises(ValueError, match="non-empty string"):
        Zone(zone_id="", vertices=((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)))


# ====================================================================
# Point-in-Polygon Membership & Boundary Semantics Tests
# ====================================================================

def test_point_in_convex_polygon():
    # Square: (100, 100) to (200, 200)
    square = ((100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0))
    zone = Zone(zone_id="square", vertices=square)

    # Strictly inside
    assert zone.contains_point(150.0, 150.0) is True
    assert zone.contains_point(101.0, 101.0) is True

    # Strictly outside
    assert zone.contains_point(50.0, 150.0) is False
    assert zone.contains_point(250.0, 150.0) is False
    assert zone.contains_point(150.0, 50.0) is False
    assert zone.contains_point(150.0, 250.0) is False


def test_closed_boundary_inclusive_semantics():
    """Verify points exactly on polygon edges or vertices evaluate as inside."""
    square = ((100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0))
    zone = Zone(zone_id="square", vertices=square)

    # Exactly on vertices
    assert zone.contains_point(100.0, 100.0, inclusive=True) is True
    assert zone.contains_point(200.0, 100.0, inclusive=True) is True
    assert zone.contains_point(200.0, 200.0, inclusive=True) is True
    assert zone.contains_point(100.0, 200.0, inclusive=True) is True

    # Exactly on edges
    assert zone.contains_point(150.0, 100.0, inclusive=True) is True  # Top edge
    assert zone.contains_point(200.0, 150.0, inclusive=True) is True  # Right edge
    assert zone.contains_point(150.0, 200.0, inclusive=True) is True  # Bottom edge
    assert zone.contains_point(100.0, 150.0, inclusive=True) is True  # Left edge


def test_concave_l_shaped_polygon():
    """Verify concave polygon interior and exterior cut-out queries."""
    # L-shaped polygon: 100x100 with a 50x50 cut-out at top-right
    l_shape = (
        (0.0, 0.0),
        (50.0, 0.0),
        (50.0, 50.0),
        (100.0, 50.0),
        (100.0, 100.0),
        (0.0, 100.0),
    )
    zone = Zone(zone_id="L_zone", vertices=l_shape)

    # Inside left column
    assert zone.contains_point(25.0, 25.0) is True
    assert zone.contains_point(25.0, 75.0) is True
    # Inside bottom leg
    assert zone.contains_point(75.0, 75.0) is True

    # In the cut-out corner (outside)
    assert zone.contains_point(75.0, 25.0) is False
    # Far outside
    assert zone.contains_point(150.0, 150.0) is False


# ====================================================================
# Multiple Zones & Overlap Policy Tests
# ====================================================================

def test_multiple_disjoint_zones():
    z_a = Zone(zone_id="A", vertices=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)))
    z_b = Zone(zone_id="B", vertices=((200.0, 0.0), (300.0, 0.0), (300.0, 100.0), (200.0, 100.0)))
    engine = ZoneEngine([z_a, z_b])

    # Inside A only
    m_a = engine.evaluate_point(50.0, 50.0)
    assert m_a == ("A",)

    # Inside B only
    m_b = engine.evaluate_point(250.0, 50.0)
    assert m_b == ("B",)

    # In between (outside all)
    m_none = engine.evaluate_point(150.0, 50.0)
    assert m_none == ()


def test_overlapping_zones_independent_evaluation():
    """Verify overlapping zones are evaluated independently without arbitrary tie-breaking."""
    # Zone A: (0, 0) to (150, 150)
    # Zone B: (100, 100) to (250, 250)
    z_a = Zone(zone_id="A", vertices=((0.0, 0.0), (150.0, 0.0), (150.0, 150.0), (0.0, 150.0)))
    z_b = Zone(zone_id="B", vertices=((100.0, 100.0), (250.0, 100.0), (250.0, 250.0), (100.0, 250.0)))
    engine = ZoneEngine([z_a, z_b])

    # Point in overlap region (120, 120)
    obs_overlap = FootpointObservation(track_id=1, frame_index=1, x=120.0, y=120.0)
    m = engine.evaluate_observation(obs_overlap)
    assert m.is_in_zone is True
    assert m.zone_ids == ("A", "B")
    assert m.primary_zone_id is None  # Multiple matches produce None (no arbitrary tie-breaking)

    # Point in Zone A only (50, 50)
    m_a = engine.evaluate_observation(FootpointObservation(1, 1, 50.0, 50.0))
    assert m_a.zone_ids == ("A",)
    assert m_a.primary_zone_id == "A"

    # Point in Zone B only (200, 200)
    m_b = engine.evaluate_observation(FootpointObservation(1, 1, 200.0, 200.0))
    assert m_b.zone_ids == ("B",)
    assert m_b.primary_zone_id == "B"


def test_primary_zone_id_semantics():
    """Verify primary_zone_id returns single zone ID or None for 0 or multiple matches."""
    # Zero matched zones
    m_zero = ZoneMembership(track_id=1, frame_index=1, x=10.0, y=10.0, zone_ids=())
    assert m_zero.primary_zone_id is None
    assert m_zero.is_in_zone is False

    # Exactly one matched zone
    m_one = ZoneMembership(track_id=1, frame_index=1, x=10.0, y=10.0, zone_ids=("Zone1",))
    assert m_one.primary_zone_id == "Zone1"
    assert m_one.is_in_zone is True

    # Overlapping two zones -> None
    m_two = ZoneMembership(track_id=1, frame_index=1, x=10.0, y=10.0, zone_ids=("Zone1", "Zone2"))
    assert m_two.primary_zone_id is None
    assert m_two.is_in_zone is True

    # Overlapping three zones -> None
    m_three = ZoneMembership(track_id=1, frame_index=1, x=10.0, y=10.0, zone_ids=("Zone1", "Zone2", "Zone3"))
    assert m_three.primary_zone_id is None
    assert m_three.is_in_zone is True


def test_inclusive_parameter_semantics():
    """Verify inclusive=True and inclusive=False across interior, edges, vertices, exterior."""
    square = ((100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0))
    zone = Zone(zone_id="square", vertices=square)

    # 1. Interior -> True for both
    assert zone.contains_point(150.0, 150.0, inclusive=True) is True
    assert zone.contains_point(150.0, 150.0, inclusive=False) is True
    assert zone.contains_point(105.0, 195.0, inclusive=True) is True
    assert zone.contains_point(105.0, 195.0, inclusive=False) is True

    # 2. Boundary edges -> True for inclusive=True, False for inclusive=False
    edges = [
        (150.0, 100.0),  # Top edge
        (200.0, 150.0),  # Right edge
        (150.0, 200.0),  # Bottom edge
        (100.0, 150.0),  # Left edge
    ]
    for pt in edges:
        assert zone.contains_point(pt[0], pt[1], inclusive=True) is True
        assert zone.contains_point(pt[0], pt[1], inclusive=False) is False

    # 3. Vertices -> True for inclusive=True, False for inclusive=False
    vertices = [
        (100.0, 100.0),  # Top-left vertex
        (200.0, 100.0),  # Top-right vertex
        (200.0, 200.0),  # Bottom-right vertex
        (100.0, 200.0),  # Bottom-left vertex
    ]
    for pt in vertices:
        assert zone.contains_point(pt[0], pt[1], inclusive=True) is True
        assert zone.contains_point(pt[0], pt[1], inclusive=False) is False

    # 4. Exterior -> False for both
    exterior_points = [
        (50.0, 150.0),
        (250.0, 150.0),
        (150.0, 50.0),
        (150.0, 250.0),
        (50.0, 50.0),
    ]
    for pt in exterior_points:
        assert zone.contains_point(pt[0], pt[1], inclusive=True) is False
        assert zone.contains_point(pt[0], pt[1], inclusive=False) is False

    # Default parameter must be inclusive=True
    assert zone.contains_point(150.0, 100.0) is True
    assert zone.contains_point(100.0, 100.0) is True


def test_deterministic_zone_ids_ordering():
    """Verify zone_ids are sorted deterministically independent of registration order."""
    # Three identical spatial zones with different IDs
    verts = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    z_alpha = Zone("Alpha", verts)
    z_beta = Zone("Beta", verts)
    z_gamma = Zone("Gamma", verts)

    # Register in order: Gamma, Beta, Alpha
    engine1 = ZoneEngine([z_gamma, z_beta, z_alpha])
    res1 = engine1.evaluate_point(50.0, 50.0)
    assert res1 == ("Alpha", "Beta", "Gamma")

    # Register in reverse order: Beta, Alpha, Gamma
    engine2 = ZoneEngine([z_beta, z_alpha, z_gamma])
    res2 = engine2.evaluate_point(50.0, 50.0)
    assert res2 == ("Alpha", "Beta", "Gamma")

    # Register in order: Alpha, Gamma, Beta
    engine3 = ZoneEngine([z_alpha, z_gamma, z_beta])
    res3 = engine3.evaluate_point(50.0, 50.0)
    assert res3 == ("Alpha", "Beta", "Gamma")


def test_zone_engine_duplicate_id_raises_value_error():
    z1 = Zone(zone_id="z1", vertices=((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)))
    z2 = Zone(zone_id="z1", vertices=((10.0, 10.0), (20.0, 10.0), (10.0, 20.0)))
    engine = ZoneEngine([z1])
    with pytest.raises(ValueError, match="already registered"):
        engine.add_zone(z2)


def test_zone_engine_get_and_remove():
    z1 = Zone(zone_id="z1", vertices=((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)))
    engine = ZoneEngine([z1])
    assert engine.get_zone("z1") == z1

    with pytest.raises(KeyError, match="not found"):
        engine.get_zone("unknown")

    engine.remove_zone("z1")
    assert engine.zone_ids == []


# ====================================================================
# Trajectory & Gap Preservation Tests
# ====================================================================

def test_evaluate_trajectory_preserves_gaps():
    """Verify trajectory evaluations produce memberships ONLY for observed frames."""
    zone_a = Zone("A", ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)))
    zone_b = Zone("B", ((200.0, 0.0), (300.0, 0.0), (300.0, 100.0), (200.0, 100.0)))
    engine = ZoneEngine([zone_a, zone_b])

    # Trajectory with gap at frames 12 and 13
    points = (
        FootpointObservation(track_id=1, frame_index=10, x=50.0, y=50.0),   # In Zone A
        FootpointObservation(track_id=1, frame_index=11, x=60.0, y=50.0),   # In Zone A
        FootpointObservation(track_id=1, frame_index=14, x=250.0, y=50.0),  # In Zone B
    )
    traj = Trajectory(track_id=1, points=points)
    assert traj.has_gaps is True

    memberships = engine.evaluate_trajectory(traj)

    # Exactly 3 membership observations; no interpolated frames 12/13
    assert len(memberships) == 3
    assert memberships[0].frame_index == 10
    assert memberships[0].zone_ids == ("A",)

    assert memberships[1].frame_index == 11
    assert memberships[1].zone_ids == ("A",)

    assert memberships[2].frame_index == 14
    assert memberships[2].zone_ids == ("B",)


# ====================================================================
# Visualization Tests
# ====================================================================

def test_draw_zones_does_not_mutate_image():
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    canvas_copy = canvas.copy()

    zone = Zone(zone_id="Z1", vertices=((50.0, 50.0), (200.0, 50.0), (200.0, 200.0), (50.0, 200.0)))
    m = ZoneMembership(track_id=1, frame_index=1, x=100.0, y=100.0, zone_ids=("Z1",))

    output = draw_zones(canvas, [zone], memberships=[m])
    assert output.shape == canvas.shape
    assert output.dtype == canvas.dtype
    # Canvas must not be modified in place
    assert np.array_equal(canvas, canvas_copy)
    # Output must have rendered content
    assert not np.array_equal(output, canvas)
