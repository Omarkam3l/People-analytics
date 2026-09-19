# People Movement Analytics

A Computer Vision system for analyzing human movement in fixed-camera video.

The system detects and tracks people, extracts their footpoints, builds movement trajectories, generates spatial heatmaps, and calculates dwell time inside defined regions.

The project is designed as an end-to-end Computer Vision and spatial analytics project, with an emphasis on understanding the underlying geometry and algorithms rather than treating the system as a black box.

---

## Core Pipeline

Video
  ↓
Person Detection
  ↓
Multi-Object Tracking
  ↓
Bounding Boxes
  ↓
Footpoint Extraction
  ↓
Trajectory Generation
  ↓
┌───────────────────┐
│                   │
▼                   ▼
Heatmap          Dwell Time
│                   │
└─────────┬─────────┘
          ▼
     Spatial Analytics

Future extension:

Footpoint
    ↓
Homography
    ↓
Ground-Plane Coordinates
    ↓
Real-World Spatial Analytics

---

## Main Goals

The system should:

1. Detect people in video.
2. Track each person across frames.
3. Maintain a stable identity (`track_id`) for each tracked person.
4. Extract a footpoint from each person's bounding box.
5. Build trajectories from sequential footpoints.
6. Generate spatial heatmaps.
7. Define spatial zones.
8. Calculate how long each tracked person remains inside each zone.
9. Produce aggregate dwell-time statistics.
10. Evaluate the quality of the system using ground-truth annotations where available.

---

## Dataset

The initial dataset is MOT17 from the MOTChallenge benchmark.

MOT17 is used as the primary dataset for:

- Person detection
- Multi-object tracking
- Trajectory generation
- Footpoint extraction
- Heatmap generation
- Dwell-time experiments
- Evaluation

The project does not initially require a custom dataset.

A future version may use datasets containing camera calibration and world-coordinate ground truth for accurate ground-plane projection.

---

## Important Concepts

This project combines several Computer Vision concepts:

- Object Detection
- Multi-Object Tracking
- Bounding Boxes
- Footpoint Localization
- Coordinate Systems
- Perspective Geometry
- Homography
- Ground-Plane Projection
- Trajectory Analysis
- Spatial Density
- Heatmaps
- Zone Analysis
- Dwell Time
- Evaluation Metrics

---

## Project Philosophy

The project should prioritize understanding and correctness over unnecessary complexity.

A pretrained detector may be used initially.

Training a custom detector is optional and should only be introduced when there is a clear reason to do so.

The first implementation should establish a correct end-to-end pipeline before introducing model fine-tuning, multi-camera fusion, or advanced 3D geometry.

---

## Current Scope

### In Scope

- MOT17
- Person detection
- Single-camera tracking
- Track IDs
- Footpoint extraction
- Trajectories
- Heatmaps
- Zone definition
- Dwell-time calculation
- Basic evaluation
- Visualization

### Future Scope

- Custom detector training
- Homography-based ground-plane projection
- Real-world coordinates
- Multi-camera tracking
- Cross-camera identity association
- Advanced spatial analytics
- Real-time processing
- Web dashboard

---

## Non-Goals for the Initial Version

The initial version will NOT attempt to:

- Build a complete production surveillance platform.
- Solve multi-camera identity association.
- Train a detector from scratch.
- Build a web application.
- Implement 3D reconstruction.
- Optimize for real-time performance before correctness is established.

---

## Expected Output

The system should eventually produce:

- Annotated video
- Bounding boxes
- Track IDs
- Footpoints
- Trajectory visualization
- Heatmap
- Zone occupancy
- Per-person dwell time
- Per-zone dwell statistics
- Evaluation metrics
- Machine-readable analytics output

---

## Development Workflow

The project follows:

Specification
→ Implementation
→ Tests
→ Engineering Audit
→ Targeted Hardening
→ API Freeze
→ Next Phase

Architecture should remain stable between approved phases unless an audit identifies a concrete architectural problem.