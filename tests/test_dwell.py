"""Unit and integration tests for Phase 8 dwell time and visit analytics."""

import numpy as np
import pytest

from people_analytics.dwell import (
    DwellConfig,
    DwellTimeEngine,
    PersonVisitSummary,
    ZoneAnalytics,
    ZoneVisit,
    draw_dwell_overlay,
)
from people_analytics.footpoint.models import FootpointObservation
from people_analytics.trajectory.models import Trajectory
from people_analytics.zone.engine import ZoneEngine
from people_analytics.zone.models import Zone, ZoneMembership


# ====================================================================
# Configuration & Invariant Tests
# ====================================================================

def test_dwell_config_validation():
    config = DwellConfig(fps=30.0, max_gap_frames=2)
    assert config.fps == 30.0
    assert config.max_gap_frames == 2

    # Default max_gap_frames is 0 (strict observation)
    strict = DwellConfig(fps=25.0)
    assert strict.max_gap_frames == 0

    with pytest.raises(ValueError, match="strictly positive"):
        DwellConfig(fps=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        DwellConfig(fps=-10.0)

    with pytest.raises(ValueError, match="strictly positive"):
        DwellConfig(fps=float("nan"))

    with pytest.raises(ValueError, match="non-negative"):
        DwellConfig(fps=30.0, max_gap_frames=-1)


def test_zone_visit_invariants():
    # Valid multi-frame visit
    v = ZoneVisit(
        track_id=1,
        zone_id="ZoneA",
        entry_frame=10,
        last_observed_frame=15,
        entry_timestamp=0.3,
        exit_timestamp=0.5,
        duration_seconds=0.2,
        observation_count=6,
    )
    assert v.track_id == 1
    assert v.duration_seconds == 0.2

    # Invalid empty zone_id
    with pytest.raises(ValueError, match="non-empty string"):
        ZoneVisit(1, "", 10, 15, 0.3, 0.5, 0.2, 6)

    # Invalid entry_frame < 1
    with pytest.raises(ValueError, match="entry_frame must be >= 1"):
        ZoneVisit(1, "A", 0, 5, 0.0, 0.5, 0.5, 5)

    # Invalid last_observed_frame < entry_frame
    with pytest.raises(ValueError, match="cannot precede"):
        ZoneVisit(1, "A", 15, 10, 0.5, 0.3, 0.0, 1)

    # Invalid observation_count < 1
    with pytest.raises(ValueError, match="observation_count must be >= 1"):
        ZoneVisit(1, "A", 10, 15, 0.3, 0.5, 0.2, 0)

    # Invalid observation_count exceeding span (span is 15-10+1 = 6)
    with pytest.raises(ValueError, match="exceeds frame span"):
        ZoneVisit(1, "A", 10, 15, 0.3, 0.5, 0.2, 7)


# ====================================================================
# Visit Lifecycle & Duration Semantics Tests
# ====================================================================

def test_single_frame_visit():
    """Verify single-frame visit has duration 0.0s and observation_count 1."""
    config = DwellConfig(fps=30.0)
    engine = DwellTimeEngine(config)

    # Single observation at frame 10 in Zone A
    m = ZoneMembership(track_id=1, frame_index=10, x=50.0, y=50.0, zone_ids=("A",))
    engine.add_membership(m)
    engine.finalize_all()

    visits = engine.visits
    assert len(visits) == 1
    v = visits[0]
    assert v.track_id == 1
    assert v.zone_id == "A"
    assert v.entry_frame == 10
    assert v.last_observed_frame == 10
    assert v.entry_timestamp == (10 - 1) / 30.0
    assert v.exit_timestamp == (10 - 1) / 30.0
    assert v.duration_seconds == 0.0
    assert v.observation_count == 1


def test_multi_frame_continuous_dwell():
    """Verify continuous dwell across consecutive frames."""
    fps = 20.0
    config = DwellConfig(fps=fps)
    engine = DwellTimeEngine(config)

    # Frames 10 through 15 in Zone A
    memberships = [
        ZoneMembership(track_id=1, frame_index=f, x=50.0, y=50.0, zone_ids=("A",))
        for f in range(10, 16)  # 10, 11, 12, 13, 14, 15 (6 frames)
    ]
    engine.add_memberships(memberships)
    engine.finalize_all()

    visits = engine.visits
    assert len(visits) == 1
    v = visits[0]
    assert v.track_id == 1
    assert v.zone_id == "A"
    assert v.entry_frame == 10
    assert v.last_observed_frame == 15
    assert v.entry_timestamp == (10 - 1) / fps
    assert v.exit_timestamp == (15 - 1) / fps
    assert pytest.approx(v.duration_seconds) == (15 - 10) / fps  # 5 / 20 = 0.25s
    assert v.observation_count == 6


def test_zone_transition():
    """Verify transition from Zone A to Zone B produces two distinct visits."""
    fps = 10.0
    config = DwellConfig(fps=fps)
    engine = DwellTimeEngine(config)

    memberships = [
        ZoneMembership(track_id=1, frame_index=10, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=11, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=12, x=150.0, y=150.0, zone_ids=("B",)),
        ZoneMembership(track_id=1, frame_index=13, x=150.0, y=150.0, zone_ids=("B",)),
    ]
    engine.add_memberships(memberships)
    engine.finalize_all()

    visits = engine.visits
    assert len(visits) == 2

    # Visit 1: Zone A
    v_a = visits[0]
    assert v_a.zone_id == "A"
    assert v_a.entry_frame == 10
    assert v_a.last_observed_frame == 11
    assert pytest.approx(v_a.duration_seconds) == 1.0 / fps
    assert v_a.observation_count == 2

    # Visit 2: Zone B
    v_b = visits[1]
    assert v_b.zone_id == "B"
    assert v_b.entry_frame == 12
    assert v_b.last_observed_frame == 13
    assert pytest.approx(v_b.duration_seconds) == 1.0 / fps
    assert v_b.observation_count == 2


def test_multiple_visits_to_same_zone():
    """Verify non-contiguous visits to the same zone form separate visit records."""
    config = DwellConfig(fps=10.0)
    engine = DwellTimeEngine(config)

    memberships = [
        # Visit 1 to Zone A: frames 10..12
        ZoneMembership(track_id=1, frame_index=10, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=11, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=12, x=50.0, y=50.0, zone_ids=("A",)),
        # Outside zone: frames 13..15
        ZoneMembership(track_id=1, frame_index=13, x=500.0, y=500.0, zone_ids=()),
        ZoneMembership(track_id=1, frame_index=14, x=500.0, y=500.0, zone_ids=()),
        ZoneMembership(track_id=1, frame_index=15, x=500.0, y=500.0, zone_ids=()),
        # Visit 2 to Zone A: frames 16..18
        ZoneMembership(track_id=1, frame_index=16, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=17, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=18, x=50.0, y=50.0, zone_ids=("A",)),
    ]
    engine.add_memberships(memberships)
    engine.finalize_all()

    visits = engine.visits
    assert len(visits) == 2
    assert visits[0].entry_frame == 10
    assert visits[0].last_observed_frame == 12
    assert visits[1].entry_frame == 16
    assert visits[1].last_observed_frame == 18


# ====================================================================
# Gap Policy Tests
# ====================================================================

def test_gap_policy_strict_observation_default():
    """Verify max_gap_frames=0 splits visits on any missing frames."""
    config = DwellConfig(fps=10.0, max_gap_frames=0)
    engine = DwellTimeEngine(config)

    # Frame 10, 11 (in Zone A), frames 12-13 missing, frame 14 (in Zone A)
    memberships = [
        ZoneMembership(track_id=1, frame_index=10, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=11, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=14, x=50.0, y=50.0, zone_ids=("A",)),
    ]
    engine.add_memberships(memberships)
    engine.finalize_all()

    visits = engine.visits
    assert len(visits) == 2
    assert visits[0].entry_frame == 10
    assert visits[0].last_observed_frame == 11
    assert visits[0].observation_count == 2

    assert visits[1].entry_frame == 14
    assert visits[1].last_observed_frame == 14
    assert visits[1].observation_count == 1


def test_gap_policy_bounded_gap_tolerance():
    """Verify max_gap_frames=2 maintains visit across missing frames without counting gap frames."""
    fps = 10.0
    config = DwellConfig(fps=fps, max_gap_frames=2)
    engine = DwellTimeEngine(config)

    # Frame 10, 11 (in Zone A), frames 12-13 missing (gap = 2 frames), frame 14 (in Zone A)
    memberships = [
        ZoneMembership(track_id=1, frame_index=10, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=11, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=14, x=50.0, y=50.0, zone_ids=("A",)),
    ]
    engine.add_memberships(memberships)
    engine.finalize_all()

    visits = engine.visits
    assert len(visits) == 1
    v = visits[0]
    assert v.entry_frame == 10
    assert v.last_observed_frame == 14
    # duration covers 10 to 14: (14 - 10) / 10.0 = 0.4s
    assert pytest.approx(v.duration_seconds) == 0.4
    # Exactly 3 actual observations; frames 12 and 13 are NOT counted
    assert v.observation_count == 3


def test_gap_policy_exceeded_tolerance():
    """Verify gap exceeding max_gap_frames terminates the visit."""
    config = DwellConfig(fps=10.0, max_gap_frames=1)
    engine = DwellTimeEngine(config)

    # Gap of 2 frames (12, 13) exceeds tolerance of 1
    memberships = [
        ZoneMembership(track_id=1, frame_index=10, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=11, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=14, x=50.0, y=50.0, zone_ids=("A",)),
    ]
    engine.add_memberships(memberships)
    engine.finalize_all()

    visits = engine.visits
    assert len(visits) == 2
    assert visits[0].last_observed_frame == 11
    assert visits[1].entry_frame == 14


# ====================================================================
# Overlapping Zones & Multi-Person Tests
# ====================================================================

def test_overlapping_zones_independent_tracking():
    """Verify overlapping zones are tracked independently per person."""
    fps = 10.0
    config = DwellConfig(fps=fps)
    engine = DwellTimeEngine(config)

    # Frame 10: ("A", "B")
    # Frame 11: ("A", "B")
    # Frame 12: ("B",)  (Exits A, stays in B)
    # Frame 13: ()      (Exits B)
    memberships = [
        ZoneMembership(track_id=1, frame_index=10, x=50.0, y=50.0, zone_ids=("A", "B")),
        ZoneMembership(track_id=1, frame_index=11, x=50.0, y=50.0, zone_ids=("A", "B")),
        ZoneMembership(track_id=1, frame_index=12, x=50.0, y=50.0, zone_ids=("B",)),
        ZoneMembership(track_id=1, frame_index=13, x=500.0, y=500.0, zone_ids=()),
    ]
    engine.add_memberships(memberships)
    engine.finalize_all()

    visits = engine.visits
    assert len(visits) == 2

    v_a = next(v for v in visits if v.zone_id == "A")
    assert v_a.entry_frame == 10
    assert v_a.last_observed_frame == 11
    assert pytest.approx(v_a.duration_seconds) == 1.0 / fps
    assert v_a.observation_count == 2

    v_b = next(v for v in visits if v.zone_id == "B")
    assert v_b.entry_frame == 10
    assert v_b.last_observed_frame == 12
    assert pytest.approx(v_b.duration_seconds) == 2.0 / fps
    assert v_b.observation_count == 3


def test_multiple_simultaneous_tracks():
    """Verify multiple distinct track IDs in the same zone maintain independent visits."""
    config = DwellConfig(fps=10.0)
    engine = DwellTimeEngine(config)

    memberships = [
        # Track 1: frames 10..12
        ZoneMembership(track_id=1, frame_index=10, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=11, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=12, x=50.0, y=50.0, zone_ids=("A",)),
        # Track 2: frames 11..15
        ZoneMembership(track_id=2, frame_index=11, x=60.0, y=60.0, zone_ids=("A",)),
        ZoneMembership(track_id=2, frame_index=12, x=60.0, y=60.0, zone_ids=("A",)),
        ZoneMembership(track_id=2, frame_index=13, x=60.0, y=60.0, zone_ids=("A",)),
        ZoneMembership(track_id=2, frame_index=14, x=60.0, y=60.0, zone_ids=("A",)),
        ZoneMembership(track_id=2, frame_index=15, x=60.0, y=60.0, zone_ids=("A",)),
    ]
    engine.add_memberships(memberships)
    engine.finalize_all()

    visits = engine.visits
    assert len(visits) == 2

    v1 = engine.get_visits_for_track(1)[0]
    assert v1.entry_frame == 10
    assert v1.last_observed_frame == 12
    assert v1.observation_count == 3

    v2 = engine.get_visits_for_track(2)[0]
    assert v2.entry_frame == 11
    assert v2.last_observed_frame == 15
    assert v2.observation_count == 5


def test_track_termination_inside_zone():
    """Verify track termination correctly finalizes open visit at last observed frame."""
    config = DwellConfig(fps=10.0)
    engine = DwellTimeEngine(config)

    # Track enters zone and sequence terminates after frame 12
    memberships = [
        ZoneMembership(track_id=1, frame_index=10, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=11, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=12, x=50.0, y=50.0, zone_ids=("A",)),
    ]
    engine.add_memberships(memberships)
    engine.finalize_all()

    visits = engine.visits
    assert len(visits) == 1
    assert visits[0].last_observed_frame == 12
    assert visits[0].observation_count == 3


def test_unsorted_batch_ingestion():
    """Verify builder normalizes out-of-order frame observations."""
    config = DwellConfig(fps=10.0)
    engine = DwellTimeEngine(config)

    # Ingest frames out of order: 12, 10, 11
    memberships = [
        ZoneMembership(track_id=1, frame_index=12, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=10, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=11, x=50.0, y=50.0, zone_ids=("A",)),
    ]
    engine.add_memberships(memberships)
    engine.finalize_all()

    visits = engine.visits
    assert len(visits) == 1
    assert visits[0].entry_frame == 10
    assert visits[0].last_observed_frame == 12
    assert visits[0].observation_count == 3


def test_duplicate_frame_raises_value_error():
    """Verify duplicate observation for the same (track_id, frame_index) raises ValueError."""
    config = DwellConfig(fps=10.0)
    engine = DwellTimeEngine(config)

    engine.add_membership(ZoneMembership(track_id=1, frame_index=10, x=50.0, y=50.0, zone_ids=("A",)))
    engine.add_membership(ZoneMembership(track_id=1, frame_index=10, x=60.0, y=60.0, zone_ids=("A",)))

    with pytest.raises(ValueError, match="Duplicate observation for track_id 1 at frame 10"):
        engine.finalize_all()


# ====================================================================
# Aggregate Statistics Tests
# ====================================================================

def test_zone_analytics_computation():
    fps = 10.0
    config = DwellConfig(fps=fps)
    engine = DwellTimeEngine(config)

    # Track 1 visits Zone A: frames 10..15 (6 frames, duration = 0.5s)
    # Track 2 visits Zone A: frames 20..22 (3 frames, duration = 0.2s)
    # Track 1 visits Zone A again: frames 30..30 (1 frame, duration = 0.0s)
    memberships = [
        # Track 1 visit 1
        ZoneMembership(1, 10, 50.0, 50.0, ("A",)),
        ZoneMembership(1, 15, 50.0, 50.0, ("A",)),
        # Track 2 visit
        ZoneMembership(2, 20, 50.0, 50.0, ("A",)),
        ZoneMembership(2, 22, 50.0, 50.0, ("A",)),
        # Track 1 visit 2
        ZoneMembership(1, 30, 50.0, 50.0, ("A",)),
    ]
    # Use max_gap_frames=10 so Track 1's 10..15 is 1 visit
    config = DwellConfig(fps=fps, max_gap_frames=10)
    engine = DwellTimeEngine(config)
    engine.add_memberships(memberships)
    engine.finalize_all()

    analytics = engine.compute_zone_analytics("A")
    assert analytics.zone_id == "A"
    assert analytics.total_visits == 3
    assert analytics.unique_visitors == 2  # Track 1 and Track 2
    assert pytest.approx(analytics.total_dwell_time) == 0.5 + 0.2 + 0.0
    assert pytest.approx(analytics.average_dwell_time) == (0.7) / 3
    assert pytest.approx(analytics.max_dwell_time) == 0.5
    assert analytics.total_observations == 5

    # Compute for empty zone with zero visits
    empty_analytics = engine.compute_zone_analytics("EmptyZone")
    assert empty_analytics.total_visits == 0
    assert empty_analytics.unique_visitors == 0
    assert empty_analytics.total_dwell_time == 0.0
    assert empty_analytics.average_dwell_time == 0.0
    assert empty_analytics.max_dwell_time == 0.0
    assert empty_analytics.total_observations == 0

    # Test compute_all_zone_analytics
    all_analytics = engine.compute_all_zone_analytics()
    assert "A" in all_analytics


def test_person_visit_summary():
    fps = 10.0
    config = DwellConfig(fps=fps)
    engine = DwellTimeEngine(config)

    memberships = [
        # Track 1 in Zone A: frames 10..12 (duration = 0.2s)
        ZoneMembership(1, 10, 50.0, 50.0, ("A",)),
        ZoneMembership(1, 11, 50.0, 50.0, ("A",)),
        ZoneMembership(1, 12, 50.0, 50.0, ("A",)),
        # Track 1 in Zone B: frames 13..15 (duration = 0.2s)
        ZoneMembership(1, 13, 150.0, 150.0, ("B",)),
        ZoneMembership(1, 14, 150.0, 150.0, ("B",)),
        ZoneMembership(1, 15, 150.0, 150.0, ("B",)),
    ]
    engine.add_memberships(memberships)
    engine.finalize_all()

    summary = engine.compute_person_summary(1)
    assert summary.track_id == 1
    assert len(summary.visits) == 2
    assert pytest.approx(summary.total_dwell_time) == 0.4
    assert summary.zones_visited == ("A", "B")

    # Person with no visits
    empty_summary = engine.compute_person_summary(999)
    assert empty_summary.track_id == 999
    assert empty_summary.visits == ()
    assert empty_summary.total_dwell_time == 0.0
    assert empty_summary.zones_visited == ()


# ====================================================================
# Pipeline Integration & Visualization Tests
# ====================================================================

def test_pipeline_trajectory_to_dwell_engine():
    """Verify integration from Trajectory through ZoneEngine to DwellTimeEngine."""
    # Define a spatial zone: (0, 0) to (100, 100)
    zone_a = Zone(zone_id="ZoneA", vertices=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)))
    zone_engine = ZoneEngine([zone_a])

    # Trajectory enters Zone A at frame 10, leaves at frame 13
    points = (
        FootpointObservation(track_id=1, frame_index=10, x=50.0, y=50.0),
        FootpointObservation(track_id=1, frame_index=11, x=50.0, y=50.0),
        FootpointObservation(track_id=1, frame_index=12, x=50.0, y=50.0),
        FootpointObservation(track_id=1, frame_index=13, x=500.0, y=500.0),  # Outside
    )
    traj = Trajectory(track_id=1, points=points)

    dwell_engine = DwellTimeEngine(DwellConfig(fps=10.0))
    dwell_engine.add_trajectory(traj, zone_engine)
    dwell_engine.finalize_all()

    visits = dwell_engine.visits
    assert len(visits) == 1
    v = visits[0]
    assert v.zone_id == "ZoneA"
    assert v.entry_frame == 10
    assert v.last_observed_frame == 12
    assert pytest.approx(v.duration_seconds) == 0.2
    assert v.observation_count == 3


def test_draw_dwell_overlay_does_not_mutate():
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    canvas_copy = canvas.copy()

    zone = Zone(zone_id="Z1", vertices=((50.0, 50.0), (200.0, 50.0), (200.0, 200.0), (50.0, 200.0)))
    visit = ZoneVisit(
        track_id=1,
        zone_id="Z1",
        entry_frame=10,
        last_observed_frame=20,
        entry_timestamp=0.3,
        exit_timestamp=0.6,
        duration_seconds=0.3,
        observation_count=11,
    )

    output = draw_dwell_overlay(canvas, [zone], [visit], current_frame=15)
    assert output.shape == canvas.shape
    assert output.dtype == canvas.dtype
    assert np.array_equal(canvas, canvas_copy)
    assert not np.array_equal(output, canvas)


# ====================================================================
# Final Semantic Hardening & Boundary Tests
# ====================================================================

def test_gap_boundary_max_gap_zero():
    """Boundary test for max_gap_frames=0 (strict observation).

    Definition: missing_frames = current_frame - last_observed_frame - 1.
    - 0 missing frames (frames 10, 11) -> visit continues.
    - 1 missing frame (frames 10, 12) -> visit terminates and splits.
    """
    fps = 10.0
    config = DwellConfig(fps=fps, max_gap_frames=0)

    # 1. Contiguous: missing_frames = 0
    engine1 = DwellTimeEngine(config)
    engine1.add_membership(ZoneMembership(1, 10, 50.0, 50.0, ("A",)))
    engine1.add_membership(ZoneMembership(1, 11, 50.0, 50.0, ("A",)))
    visits1 = engine1.visits
    assert len(visits1) == 1
    assert visits1[0].entry_frame == 10
    assert visits1[0].last_observed_frame == 11
    assert visits1[0].observation_count == 2
    assert pytest.approx(visits1[0].duration_seconds) == 0.1

    # 2. One missing frame: missing_frames = 12 - 10 - 1 = 1 > 0
    engine2 = DwellTimeEngine(config)
    engine2.add_membership(ZoneMembership(1, 10, 50.0, 50.0, ("A",)))
    engine2.add_membership(ZoneMembership(1, 12, 50.0, 50.0, ("A",)))
    visits2 = engine2.visits
    assert len(visits2) == 2
    assert visits2[0].entry_frame == 10
    assert visits2[0].last_observed_frame == 10
    assert visits2[0].observation_count == 1
    assert visits2[0].duration_seconds == 0.0

    assert visits2[1].entry_frame == 12
    assert visits2[1].last_observed_frame == 12
    assert visits2[1].observation_count == 1
    assert visits2[1].duration_seconds == 0.0


def test_gap_boundary_exact_and_beyond_k():
    """Boundary test around max_gap_frames=K (tolerance boundary).

    With max_gap_frames = 2:
    - Exactly allowed gap: missing_frames = 13 - 10 - 1 = 2 == K -> visit survives.
    - One frame beyond allowed: missing_frames = 14 - 10 - 1 = 3 > K -> visit splits.
    """
    fps = 10.0
    config = DwellConfig(fps=fps, max_gap_frames=2)

    # 1. Exactly allowed gap (2 missing frames: 11, 12)
    engine1 = DwellTimeEngine(config)
    engine1.add_membership(ZoneMembership(1, 10, 50.0, 50.0, ("A",)))
    engine1.add_membership(ZoneMembership(1, 13, 50.0, 50.0, ("A",)))
    visits1 = engine1.visits
    assert len(visits1) == 1
    assert visits1[0].entry_frame == 10
    assert visits1[0].last_observed_frame == 13
    assert visits1[0].observation_count == 2
    assert pytest.approx(visits1[0].duration_seconds) == 0.3  # (13 - 10) / 10

    # 2. One frame beyond allowed gap (3 missing frames: 11, 12, 13)
    engine2 = DwellTimeEngine(config)
    engine2.add_membership(ZoneMembership(1, 10, 50.0, 50.0, ("A",)))
    engine2.add_membership(ZoneMembership(1, 14, 50.0, 50.0, ("A",)))
    visits2 = engine2.visits
    assert len(visits2) == 2
    assert visits2[0].entry_frame == 10
    assert visits2[0].last_observed_frame == 10
    assert visits2[0].observation_count == 1
    assert visits2[0].duration_seconds == 0.0

    assert visits2[1].entry_frame == 14
    assert visits2[1].last_observed_frame == 14
    assert visits2[1].observation_count == 1
    assert visits2[1].duration_seconds == 0.0


def test_single_frame_transition_and_isolated_durations():
    """Verify exact 0.0s duration semantics for 1-observation contacts in transitions and isolation."""
    fps = 30.0
    config = DwellConfig(fps=fps)
    engine = DwellTimeEngine(config)

    # Track 1: Frame 10 -> Zone A, Frame 11 -> Zone B (instantaneous 1-frame visits in each)
    engine.add_membership(ZoneMembership(1, 10, 50.0, 50.0, ("A",)))
    engine.add_membership(ZoneMembership(1, 11, 150.0, 150.0, ("B",)))
    # Track 2: Frame 10 -> Zone A only
    engine.add_membership(ZoneMembership(2, 10, 50.0, 50.0, ("A",)))
    engine.finalize_all()

    # Track 1 visits
    t1_visits = engine.get_visits_for_track(1)
    assert len(t1_visits) == 2
    assert t1_visits[0].zone_id == "A"
    assert t1_visits[0].entry_frame == 10
    assert t1_visits[0].last_observed_frame == 10
    assert t1_visits[0].duration_seconds == 0.0
    assert t1_visits[0].observation_count == 1

    assert t1_visits[1].zone_id == "B"
    assert t1_visits[1].entry_frame == 11
    assert t1_visits[1].last_observed_frame == 11
    assert t1_visits[1].duration_seconds == 0.0
    assert t1_visits[1].observation_count == 1

    # Track 2 visit
    t2_visits = engine.get_visits_for_track(2)
    assert len(t2_visits) == 1
    assert t2_visits[0].zone_id == "A"
    assert t2_visits[0].entry_frame == 10
    assert t2_visits[0].last_observed_frame == 10
    assert t2_visits[0].duration_seconds == 0.0
    assert t2_visits[0].observation_count == 1


def test_end_of_input_finalization():
    """Verify finalization is end-of-input finalization independent of upstream TrackState."""
    config = DwellConfig(fps=10.0)
    engine = DwellTimeEngine(config)

    # Ingest frames 10, 11, 12 without explicit termination signal
    engine.add_membership(ZoneMembership(1, 10, 50.0, 50.0, ("A",)))
    engine.add_membership(ZoneMembership(1, 11, 50.0, 50.0, ("A",)))
    engine.add_membership(ZoneMembership(1, 12, 50.0, 50.0, ("A",)))

    # Merely querying visits triggers end-of-input build and seals the visit
    visits = engine.visits
    assert len(visits) == 1
    assert visits[0].entry_frame == 10
    assert visits[0].last_observed_frame == 12
    assert visits[0].observation_count == 3
    assert pytest.approx(visits[0].duration_seconds) == 0.2


def test_overlapping_zones_independent_fsm_no_primary_winner():
    """Confirm independently maintained state per (track_id, zone_id) with zero primary_zone_id dependence."""
    fps = 10.0
    config = DwellConfig(fps=fps)
    engine = DwellTimeEngine(config)

    m10 = ZoneMembership(1, 10, 50.0, 50.0, ("A", "B"))
    m11 = ZoneMembership(1, 11, 50.0, 50.0, ("A", "B"))
    m12 = ZoneMembership(1, 12, 50.0, 50.0, ("B",))

    # In Phase 7, primary_zone_id is explicitly None for multi-zone matches
    assert m10.primary_zone_id is None
    assert m11.primary_zone_id is None
    assert m12.primary_zone_id == "B"

    engine.add_memberships([m10, m11, m12])
    engine.finalize_all()

    visits = engine.visits
    assert len(visits) == 2

    v_a = next(v for v in visits if v.zone_id == "A")
    assert v_a.entry_frame == 10
    assert v_a.last_observed_frame == 11
    assert v_a.observation_count == 2
    assert pytest.approx(v_a.duration_seconds) == 0.1

    v_b = next(v for v in visits if v.zone_id == "B")
    assert v_b.entry_frame == 10
    assert v_b.last_observed_frame == 12
    assert v_b.observation_count == 3
    assert pytest.approx(v_b.duration_seconds) == 0.2


def test_visits_property_idempotency_and_no_duplication():
    """Verify repeated access to .visits is idempotent, deterministic, and does not duplicate records."""
    fps = 10.0
    config = DwellConfig(fps=fps)
    engine = DwellTimeEngine(config)

    # Ingest observations
    engine.add_memberships([
        ZoneMembership(track_id=1, frame_index=10, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=1, frame_index=11, x=50.0, y=50.0, zone_ids=("A",)),
        ZoneMembership(track_id=2, frame_index=10, x=150.0, y=150.0, zone_ids=("B",)),
        ZoneMembership(track_id=2, frame_index=12, x=150.0, y=150.0, zone_ids=("B",)),  # Gap of 1 frame terminates at 10
    ])

    # 1. First access triggers lazy compilation
    visits_run1 = engine.visits
    assert len(visits_run1) == 3

    # 2. Repeated consecutive accesses return identical records without duplication
    for _ in range(5):
        visits_subsequent = engine.visits
        assert len(visits_subsequent) == 3
        assert visits_subsequent == visits_run1
        # Confirms defensive copy: not the same object reference
        assert visits_subsequent is not visits_run1

    # 3. Explicit finalize_all() also produces no duplication
    engine.finalize_all()
    assert len(engine.visits) == 3
    assert engine.visits == visits_run1

    # 4. Adding new observations marks engine dirty and compiles additional visits cleanly
    engine.add_membership(ZoneMembership(track_id=3, frame_index=20, x=50.0, y=50.0, zone_ids=("A",)))
    visits_after_add = engine.visits
    assert len(visits_after_add) == 4


