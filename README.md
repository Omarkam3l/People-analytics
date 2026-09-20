# People Movement Analytics

A high-integrity Computer Vision and dual-space spatial analytics system for analyzing human movement in fixed-camera video.

The system detects and tracks pedestrians, extracts bottom-center ground contact points (footpoints), constructs historical trajectories with gap preservation, accumulates spatial density heatmaps, evaluates multi-zone polygon memberships, calculates dwell times with visit state machines, and projects coordinates between camera perspective and planar bird's-eye ground coordinates via homography.

---

## Dual-Space Pipeline Architecture

```text
                           Video Frames (MOT17)
                                    │
                                    ▼
                         Person Detection (YOLOv8)
                                    │
                                    ▼
                       Multi-Object Tracking (ByteTrack)
                                    │
                                    ▼
                        Footpoint Extraction (Bottom-Center)
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         │                                                     │
         ▼ (Image Space)                                       ▼ (Planar Projection)
┌───────────────────────────────┐                     ┌─────────────────────────────────┐
│ Image Trajectory Builder      │                     │ Homography Projector            │
│ (Gap preservation, lifespan)  │                     │ (4-point planar H, condition #) │
│                               │                     └────────────────┬────────────────┘
│ Image Heatmap Accumulator     │                                      │
│ (Strict OOB rejection)        │                                      ▼ (Ground Space)
│                               │                     ┌─────────────────────────────────┐
│ Image Zone Engine             │                     │ Ground Trajectory Builder       │
│ (Ray-casting polygon tests)   │                     │ (Extrapolation policy aware)    │
│                               │                     │                                 │
│ Image Dwell Engine            │                     │ Ground Heatmap Accumulator      │
│ (Visit state machine)         │                     │ (Planar density grid)           │
└───────────────┬───────────────┘                     │                                 │
                │                                     │ Ground Zone Engine              │
                │                                     │ (Planar polygon memberships)    │
                │                                     │                                 │
                │                                     │ Ground Dwell Engine             │
                │                                     │ (Ground-plane visit tracking)   │
                │                                     └────────────────┬────────────────┘
                │                                                      │
                └───────────────────────┬──────────────────────────────┘
                                        ▼
                     Cross-Space Comparative Analytics
                     (Visit correlation & spatial agreement)
                                        │
                                        ▼
                     Mathematical Conservation Audit
                     (AC-03 to AC-07 Formal Invariants)
```

---

## Scientific Scope & Coordinate Limitation

> [!IMPORTANT]
> **Planar Coordinate Semantics**:
> Ground-plane coordinates operate strictly within `CoordinateFrame.ARBITRARY_PLANAR` using reproducible, controlled manual calibration fixtures.
> Arbitrary planar units must **NOT** be interpreted as physical meters, nor as evidence of higher physical metric accuracy. Cross-space comparison is strictly a coordinate-representation evaluation comparing perspective image-plane measurements with planar bird's-eye representations.

---

## Project Status: Phases 1–12 Completed & Frozen

| Phase | Description | Status |
| :--- | :--- | :---: |
| **Phase 1** | Project initialization, MOT17 dataset parsing, schema definitions | ✅ **Frozen** |
| **Phase 2** | YOLOv8 person detector, local greedy IoU detection evaluation | ✅ **Frozen** |
| **Phase 3** | ByteTrack multi-object tracker, CLEAR MOT & IDF1 evaluation | ✅ **Frozen** |
| **Phase 4** | Bottom-center footpoint localization and normalization | ✅ **Frozen** |
| **Phase 5** | Trajectory builder with chronological gap preservation | ✅ **Frozen** |
| **Phase 6** | Grid-based 2D heatmap accumulator with strict OOB rejection | ✅ **Frozen** |
| **Phase 7** | Spatial zone engine with ray-casting polygon containment | ✅ **Frozen** |
| **Phase 8** | Dwell time engine with per-zone visit state machines | ✅ **Frozen** |
| **Phase 9** | End-to-end integration and composite diagnostic visualizer | ✅ **Frozen** |
| **Phase 10** | Homography projection engine with ROI boundary checks | ✅ **Frozen** |
| **Phase 11** | Ground-plane movement analytics (Trajectories, Heatmaps, Zones, Dwell) | ✅ **Frozen** |
| **Phase 12** | Full-system dual-space evaluation, mathematical conservation audits, reproducibility | ✅ **Frozen** |

---

## Mathematical Conservation Invariants (AC-03 to AC-07)

Every execution is strictly validated by 7 mathematical conservation equations:

1. **Projection Conservation (AC-03)**:
   $$\text{Total Projected} \equiv \text{Valid In-ROI} + \text{Valid Extrapolated} + \text{Invalid}$$
2. **Image Heatmap Conservation (AC-04)**:
   $$\text{Total Footpoints} \equiv \text{Accumulated} + \text{Out of Bounds} + \text{Invalid}$$
3. **Ground Heatmap Conservation (AC-05)**:
   $$\text{Total Projected} \equiv \text{Accumulated} + \text{Out of Bounds} + \text{Invalid} + \text{Extrapolated Rejected}$$
4. **Image Zone Partition Conservation (AC-06a)**:
   $$\text{Total Footpoints} \equiv \text{Inside} \ge 1\text{ Zone} + \text{Outside All Zones}$$
5. **Ground Zone Partition Conservation (AC-06b)**:
   $$\text{Analytics Accepted} \equiv \text{Inside} \ge 1\text{ Zone} + \text{Outside All Zones}$$
6. **Image Dwell Overlap Conservation (AC-07a)**:
   $$\sum \text{Dwell Observations per Zone Visit} \equiv \text{Total Zone Memberships Across Zones}$$
7. **Ground Dwell Overlap Conservation (AC-07b)**:
   $$\sum \text{Ground Dwell Observations per Zone Visit} \equiv \text{Total Ground Zone Memberships Across Zones}$$

---

## Full-Sequence Benchmark Results

Evaluated across complete MOT17 benchmark sequences under frozen contracts:

| Metric Category | Metric | MOT17-09-FRCNN (Primary) | MOT17-02-FRCNN (Secondary) |
| :--- | :--- | :---: | :---: |
| **Sequence Profile** | Frames Processed | **525 / 525** (100%) | **600 / 600** (100%) |
| | Resolution & FPS | 1920×1080 @ 30.0 FPS | 1920×1080 @ 30.0 FPS |
| **Detection (Phase 2)** | True Positives (TP) | 2,154 | 3,667 |
| | Precision / Recall / F1 | 0.8875 / 0.4047 / 0.5557 | 0.8918 / 0.1974 / 0.3232 |
| **Tracking (Phase 3)** | MOTA / IDF1 | 0.2974 / 0.5401 | 0.1713 / 0.2187 |
| | ID Switches (IDSW) | 22 | 52 |
| **Trajectories** | Total Trajectories | 37 | 88 |
| | Total Observations | 3,989 | 3,975 |
| | Coverage Ratio | 0.9429 | 0.8783 |
| **Spatial Analytics** | Image Heatmap Count | 3,986 (3 OOB) | 3,974 (1 OOB) |
| | Ground Heatmap Count | 2,144 in-ROI (1,845 extrap) | 2,995 in-ROI (980 extrap) |
| | Image Dwell Time | 87.53s across 131 visits | 103.77s across 234 visits |
| | Ground Dwell Time | 54.37s across 90 visits | 96.03s across 210 visits |
| | Representation Agreement | **80.32%** (3,204 / 3,989) | **85.38%** (3,394 / 3,975) |
| **Conservation Audit** | All 7 Invariants | ✅ **ALL PASSED** | ✅ **ALL PASSED** |
| **Reproducibility** | AC-08 Verification | ✅ **REPRODUCIBLE** | ✅ **REPRODUCIBLE** |

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/Omarkam3l/People-analytics.git
cd People-analytics
pip install -e .
```

### 2. Run Full Regression Test Suite

```bash
# Complete unit and integration suite (222 tests)
python -m pytest -v

# Gated full-sequence integration suite (525/600 frames)
$env:RUN_FULL_SEQUENCE="1"; python -m pytest tests/test_full_sequence_evaluation.py -v
```

### 3. Run Full-System Evaluation CLI

```bash
# Run Primary Sequence (525 frames) with AC-08 reproducibility check and visual export:
python src/people_analytics/evaluation/run_full_evaluation.py \
  --sequence MOT17-09-FRCNN \
  --reproducibility-check \
  --visualize \
  --output-dir reports/phase12_evaluation

# Run Secondary Sequence (600 frames):
python src/people_analytics/evaluation/run_full_evaluation.py \
  --sequence MOT17-02-FRCNN \
  --visualize \
  --output-dir reports/phase12_evaluation

# Run Consolidated Multi-Sequence Evaluation:
python src/people_analytics/evaluation/run_full_evaluation.py \
  --sequence all \
  --output-dir reports/phase12_evaluation
```

---

## Evaluation Reports & Artifacts

All evaluation artifacts and visual diagnostics are located in [`reports/phase12_evaluation/`](reports/phase12_evaluation):

- [Consolidated Phase 12 Evaluation Report](reports/phase12_evaluation/PHASE_12_EVALUATION_REPORT.md)
- [MOT17-09 Evaluation Report](reports/phase12_evaluation/PHASE_12_EVALUATION_REPORT_MOT17_09.md)
- [MOT17-02 Evaluation Report](reports/phase12_evaluation/PHASE_12_EVALUATION_REPORT_MOT17_02.md)
- [Phase 12 Engineering Specification](PHASE_12_SPECIFICATION.md)
- [MOT17-09 Dual-View Composite Artifact](reports/phase12_evaluation/mot17_09_frcnn_dual_view_composite.png)
- [MOT17-09 Ground Heatmap Artifact](reports/phase12_evaluation/mot17_09_frcnn_ground_heatmap.png)
- [MOT17-02 Dual-View Composite Artifact](reports/phase12_evaluation/mot17_02_frcnn_dual_view_composite.png)
- [MOT17-02 Ground Heatmap Artifact](reports/phase12_evaluation/mot17_02_frcnn_ground_heatmap.png)

---

## Engineering Methodology

The project adheres strictly to an audited, phase-gated engineering methodology:

$$\text{Specification} \longrightarrow \text{Implementation} \longrightarrow \text{Tests} \longrightarrow \text{Engineering Audit} \longrightarrow \text{Targeted Hardening} \longrightarrow \text{API Freeze}$$

Architecture, APIs, and evaluation contracts remain frozen between approved phases.