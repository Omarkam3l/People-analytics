# Project Roadmap

## Phase 0 — Specification

Status: APPROVED

Define:

- project objective
- scope
- requirements
- coordinate conventions
- core architecture
- evaluation strategy

Deliverables:

- README.md
- SPECIFICATION.md
- ROADMAP.md


---

# Phase 1 — Dataset Understanding

Goal:

Understand MOT17 before implementing the pipeline.

Tasks:

- inspect dataset structure
- understand image/frame format
- understand annotations
- understand bounding boxes
- understand track IDs
- understand frame numbering
- understand FPS
- identify train/test sequences
- create a small dataset inspection tool

Deliverable:

A clear dataset representation and documented assumptions.

No model training yet.


---

# Phase 2 — Detection

Goal:

Produce person detections from video frames.

Tasks:

- integrate person detector
- process MOT17 frames
- generate bounding boxes
- confidence filtering
- visualize detections

Evaluation:

Compare detections against available annotations.

Deliverable:

Working person detection component.


---

# Phase 3 — Tracking

Goal:

Associate detections across frames.

Tasks:

- integrate tracker
- assign track IDs
- maintain track lifecycle
- handle lost tracks
- visualize trajectories

Evaluation:

Use MOT tracking metrics where appropriate.

Deliverable:

Stable tracked-person observations.


---

# Phase 4 — Footpoint

Goal:

Convert each tracked bounding box into a spatial footpoint.

Tasks:

- implement bottom-center calculation
- validate coordinates
- visualize footpoints
- test edge cases

Deliverable:

Reliable footpoint extraction component.


---

# Phase 5 — Trajectory

Goal:

Build time-ordered trajectories for tracked people.

Tasks:

- store observations
- preserve frame ordering
- associate footpoints with track IDs
- handle missing observations
- export trajectories

Deliverable:

Structured trajectory data.


---

# Phase 6 — Heatmap

Goal:

Generate image-space spatial heatmaps.

Tasks:

- accumulate footpoints
- configure spatial resolution
- generate density representation
- overlay heatmap on video/frame
- compare different sequences

Deliverable:

Working movement heatmap.


---

# Phase 7 — Zone Analytics

Goal:

Introduce spatial regions.

Tasks:

- define polygon zones
- implement point-in-polygon
- associate footpoints with zones
- handle zone boundaries
- visualize zones

Deliverable:

Reliable zone membership engine.


---

# Phase 8 — Dwell Time

Goal:

Calculate per-person and per-zone dwell time.

Tasks:

- detect zone entry
- detect zone exit
- calculate visit duration
- support multiple visits
- handle temporary tracking loss
- calculate aggregate statistics

Deliverable:

Dwell-time analytics.


---

# Phase 9 — Evaluation

Goal:

Validate the complete system.

Tasks:

- unit tests
- integration tests
- controlled trajectory tests
- detection evaluation
- tracking evaluation
- dwell-time validation
- heatmap validation

Deliverable:

Evaluation report and regression tests.


---

# Phase 10 — Ground-Plane Projection

Goal:

Transform image-space footpoints into ground-plane coordinates.

Tasks:

- study perspective geometry
- define calibration points
- calculate homography
- transform footpoints
- visualize bird's-eye view
- measure projection error

Deliverable:

Ground-plane spatial representation.


---

# Phase 11 — Custom Detector Training

Optional.

Only begin if the pretrained detector is insufficient.

Tasks:

- dataset preparation
- annotation validation
- training
- validation
- model comparison
- error analysis

Deliverable:

Custom detector only if justified by evaluation.


---

# Phase 12 — Advanced Analytics

Potential future work:

- speed estimation
- flow analysis
- entry/exit counting
- path analysis
- zone transition graphs
- occupancy over time
- anomaly detection
- multi-camera tracking
- real-world metric analytics
- dashboard

These are explicitly outside the initial implementation.

---

# Development Rule

Complete each phase independently.

For every phase:

Specification
→ Implementation
→ Tests
→ Engineering Audit
→ Targeted Hardening
→ API Freeze
→ Next Phase

Do not expand scope during implementation without updating the specification first.