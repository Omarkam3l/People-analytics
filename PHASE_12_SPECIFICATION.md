# Phase 12 Specification: Full-System Evaluation

**Status**: SPECIFICATION ONLY — PENDING USER REVIEW AND APPROVAL  
**Target Sequence (Primary)**: `MOT17-09-FRCNN` (Full Sequence: 525 frames, 30 FPS, 1920×1080)  
**Target Sequence (Secondary)**: `MOT17-02-FRCNN` (Full Sequence Validation: 600 frames, 30 FPS, 1920×1080)  

---

## 1. Scope

### In Scope
- **Full-Sequence Execution**: Execute the entire unified analytics pipeline across all 525 frames of `MOT17-09-FRCNN` as the primary evaluation benchmark.
- **Secondary Sequence Validation**: Execute full 600 frames of `MOT17-02-FRCNN` as secondary full-sequence verification under a distinct scene geometry.
- **Unified Dual-Space Evaluation**: Execute and quantitatively evaluate both image-space and ground-plane spatial analytics simultaneously on full sequences.
- **Engineering Conservation Invariants**: Enforce and verify all spatial conservation laws, projection invariants, and gap-honesty guarantees on complete sequences.
- **System-Level Runtime Profiling**: Capture lightweight, non-intrusive timing (elapsed time, frame throughput / FPS) and peak process memory consumption.
- **Standardized Machine and Human Outputs**: Produce structured JSON records, human-readable markdown evaluation reports, and dual-view visual diagnostic snapshots.

### Out of Scope (Strict Boundary)
- **No Architecture Redesign**: Phases 1–11 APIs and internal contracts are strictly frozen. No architectural refactoring.
- **No New Product Features**: No new domain analytics (e.g. speed, velocity vectors, entry/exit lines, graph transitions, anomaly alerts).
- **No Detector or Tracker R&D**: No fine-tuning, no custom model training, no Kalman filter redesign, and no tracker algorithmic modifications.
- **No Metric Ground Calibration**: No introduction of physical meters ($m$, $m/s$) or real-world surveying. Calibration remains strictly `CoordinateFrame.ARBITRARY_PLANAR`.
- **No UI/Dashboard Work**: No web dashboards, interactive frontends, or dynamic plotting applications.
- **No Performance Tuning / Optimization**: Profiling is purely observational; no algorithmic optimization or refactoring is permitted.

---

## 2. Pipeline Under Evaluation

The complete pipeline under evaluation executes sequentially without component omissions:

```
[Video Frames (MOT17 Sequences: 09 & 02)]
               │
               ▼
   1. Person Detection (YOLOv8PersonDetector)
               │
               ▼
   2. Multi-Object Tracking (ByteTracker)
               │
               ▼
   3. Footpoint Extraction (BottomCenterFootpointExtractor)
               │
               ▼
   4. Trajectory Construction (TrajectoryBuilder)
               │
        ┌──────┴───────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
   [Image-Space Branch]                      [Ground-Plane Branch]
   5a. Image Trajectories                     5b. Homography Projection (GroundPlaneProjector)
   6a. Image Heatmap (HeatmapAccumulator)     6b. Ground Trajectories (GroundTrajectoryBuilder)
   7a. Image Zones (ZoneEngine)               7c. Ground Heatmap (GroundHeatmapAccumulator)
   8a. Image Dwell (DwellTimeEngine)          7d. Ground Zones (GroundZoneEngine)
                                              8b. Ground Dwell (GroundDwellEngine)
        │                                              │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
            9. Dual-Space Comparison & Audit Reporting
                 (compare_image_and_ground_analytics)
```

---

## 3. Metrics and Diagnostics

### 3.1 Detection Metrics (Image Plane)
Evaluated strictly using the frozen Phase 2 / Phase 9 evaluation contract:
- **Active-Pedestrian Filtering**: Uses the exact code filtering predicate: `g.is_pedestrian and g.is_active` (`class_id == 1` and `conf == 1.0`). Non-pedestrian classes (`class_id != 1`, such as vehicles, occluders, sitting people) and inactive/ignored annotations (`conf != 1.0`) are filtered. No visibility threshold is applied by the evaluation engine.
- **Local Greedy IoU Matching**: Uses the exact local evaluation engine (`evaluate_sequence_detections`) with IoU threshold $\ge 0.5$ and greedy bipartite assignment (sorted descending by IoU).
- **Metrics**: Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$), Recall ($\frac{\text{TP}}{\text{TP} + \text{FN}}$), F1 Score ($2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$), and raw counts (Detections, TP, FP, FN).
- **Explicit Scope Limitation**: This preserves our local regression and consistency contract; it is **NOT** an official MOTChallenge detection benchmark evaluation.

### 3.2 Tracking Metrics
Evaluated strictly using the frozen Phase 3 / Phase 9 tracking evaluation contract:
- **Active-Pedestrian Filtering**: Identical to detection: filters ground truth to active pedestrians (`g.is_pedestrian and g.is_active`, i.e., `class_id == 1` and `conf == 1.0`).
- **Optimal Bipartite Matching (Hungarian)**: Uses the exact local tracking evaluation engine (`evaluate_tracking_sequence`):
  - **Frame-by-Frame CLEAR MOT Matching**: Solves optimal bipartite matching on the cost matrix $-\text{IoU}$ using `scipy.optimize.linear_sum_assignment`, accepting matches with $\text{IoU} \ge 0.5$. Unmatched tracks yield FP, unmatched GTs yield FN, and assigned track changes for a GT identity increment IDSW.
  - **Global Sequence Bipartite Matching (IDF1)**: Solves sequence-wide optimal bipartite matching (`linear_sum_assignment`) maximizing frame-level overlap counts ($\text{IoU} \ge 0.5$) between predicted track identities and ground-truth identities.
- **Metrics**: ID Switches (IDSW), MOTA ($1 - \frac{\text{FN} + \text{FP} + \text{IDSW}}{N_{\text{gt}}}$), IDF1 ($2 \cdot \frac{\text{IDTP}}{2 \cdot \text{IDTP} + \text{IDFP} + \text{IDFN}}$), Precision, Recall, Track Counts (active, confirmed, terminated), and total track observations.
- **Explicit Scope Limitation**: Preserves our local verification contract; this is **NOT** an official MOTChallenge benchmark or TrackEval submission, and reflects local YOLOv8n + ByteTracker deterministic evaluation against active pedestrian ground truth.

### 3.3 Trajectory Metrics
- **Observation Counts**: Total accumulated trajectory points across all confirmed tracks.
- **Frame Coverage**: Ratio of observed track points to the theoretical maximum span ($\text{last\_frame} - \text{first\_frame} + 1$).
- **Gap Statistics**: Total gap events ($>1$ missing frame), distribution of gap lengths, and maximum gap duration.
- **Track Lifespan**: Minimum, mean, median, and maximum track duration (in frames and seconds).

### 3.4 Image-Space Analytics Metrics
- **Heatmap Conservation**:
  $$N_{\text{footpoints}} \equiv N_{\text{heatmap accumulated}} + N_{\text{heatmap OOB}} + N_{\text{heatmap invalid}}$$
- **Zone Conservation**:
  $$N_{\text{footpoints}} \equiv N_{\text{inside at least one zone}} + N_{\text{outside all zones}}$$
- **Overlapping Zone Accounting**: Single-zone observations vs multi-zone observations.
- **Dwell Assignment Conservation**:
  $$\text{Total Dwell Observations} \equiv \sum_{o \in \text{valid}} \text{len}(o.\text{zone\_ids})$$

### 3.5 Ground-Space Analytics Metrics & Dual-Space Comparison
- **Projector Output State Conservation**:
  $$N_{\text{projected}} \equiv N_{\text{valid in-ROI}} + N_{\text{valid extrapolated}} + N_{\text{invalid}}$$
- **Downstream Analytics Acceptance (Policy Audited)**:
  - Default Policy (`allow_extrapolated=True`): Consumes $N_{\text{valid in-ROI}} + N_{\text{valid extrapolated}}$.
  - Strict Policy (`allow_extrapolated=False`): Consumes $N_{\text{valid in-ROI}}$, recording $N_{\text{valid extrapolated}}$ in `extrapolated_excluded_count`.
- **Ground Heatmap Conservation**:
  $$N_{\text{ground input}} \equiv N_{\text{accumulated}} + N_{\text{out\_of\_bounds}} + N_{\text{invalid}} + N_{\text{extrapolated\_rejected}}$$
- **Ground Zone Conservation**:
  $$N_{\text{ground obs}} \equiv N_{\text{inside at least one zone}} + N_{\text{outside all zones}}$$
- **Ground Dwell Overlap Conservation**:
  $$\text{Total Ground Dwell Assignments} \equiv \sum_{o \in \text{valid}} \text{len}(o.\text{zone\_ids})$$
- **Dual-Space Comparison Semantics**:
  - Comparison between image space and ground plane is strictly a **coordinate-representation comparison** (Image Pixel Space vs `ARBITRARY_PLANAR` coordinates).
  - It evaluates how projective perspective transformation impacts spatial clustering, density distributions, and dwell visits.
  - It must **NOT** be interpreted as demonstrating higher accuracy, physical correctness, or metric improvement, because the controlled manual calibration operates under arbitrary planar units and does not provide surveyed physical ground truth. No physical metric accuracy claim is permitted.

---

## 4. Runtime & Resource Measurements

Phase 12 will include lightweight runtime instrumentation without invasive hooks or performance optimization overhead:
- **Observational Status**: Runtime profiling is strictly **observational and descriptive**. It must **NOT** introduce pass/fail performance thresholds or latency gates in Phase 12. No performance optimization work is part of Phase 12.
- **Total Wall-Clock Time**: Measured for sequence loading, detection, tracking, footpoint extraction, image analytics, ground projection, and ground analytics.
- **Stage-by-Stage Latency Breakdown**:
  - Detection runtime (ms)
  - Tracking runtime (ms)
  - Projection & Ground Analytics runtime (ms)
- **Effective Processing FPS**: Total processed frames divided by total processing wall-clock time.
- **Process Memory Profile**: Peak Resident Set Size (RSS) in megabytes, sampled at pipeline completion using Python's standard `tracemalloc` or `psutil` if available.

---

## 5. Acceptance Criteria

Acceptance criteria are strictly invariant-based. No arbitrary numeric thresholds are imposed on YOLOv8 detection accuracy, ByteTracker MOTA scores, or processing speed/latency.

| Check ID | Verification Item | Pass/Fail Condition |
| :--- | :--- | :--- |
| **AC-01** | Full Sequence Completion | `MOT17-09-FRCNN` (525 frames) processes to completion with 0 unhandled exceptions. |
| **AC-02** | Secondary Sequence Completion | `MOT17-02-FRCNN` (600 frames) processes to completion with 0 unhandled exceptions. |
| **AC-03** | Projection Conservation | $N_{\text{projected}} = N_{\text{valid, in-ROI}} + N_{\text{valid, extrapolated}} + N_{\text{invalid}}$ holds with exact integer equality. |
| **AC-04** | Heatmap Conservation (Image) | $N_{\text{footpoints}} = N_{\text{accumulated}} + N_{\text{OOB}} + N_{\text{invalid}}$ holds exactly. |
| **AC-05** | Heatmap Conservation (Ground) | $N_{\text{ground input}} = N_{\text{accumulated}} + N_{\text{OOB}} + N_{\text{invalid}} + N_{\text{extrap\_rejected}}$ holds exactly. |
| **AC-06** | Zone Conservation (Dual) | $N_{\text{obs}} = N_{\text{inside}} + N_{\text{outside}}$ holds exactly in both image space and ground plane. |
| **AC-07** | Dwell Overlap Conservation | Total finalized dwell observation assignments $\equiv$ total zone-membership assignments across both spaces. |
| **AC-08** | Reproducibility | Analytical outputs are reproducible across consecutive runs with identical: sequence, model, configuration, calibration, and software/runtime environment.<br><br>**Requirements**:<br>• discrete counts must match exactly<br>• invariant/conservation results must match exactly<br>• floating-point aggregate metrics must match within a documented numerical tolerance (abs_tol $\le 10^{-5}$)<br>• runtime, latency, FPS, RSS, and other resource measurements are excluded from deterministic equality |
| **AC-09** | Diagnostic Traceability | Every invalid projection carries an identifiable error code (`PROJECTIVE_DENOMINATOR_TOO_SMALL`, etc.). |
| **AC-10** | Regression Invariance | All 201 pre-existing tests across Phases 1–11 pass without regression. |

---

## 6. Required Evaluation Outputs

Phase 12 execution will produce the following artifacts in `reports/phase12_evaluation/`:

1. **Machine-Readable Results (`evaluation_results_mot17_09.json` & `evaluation_results_mot17_02.json`)**:
   - Complete serialized summary of detection metrics, tracking metrics, trajectory stats, image analytics, ground analytics, timings, and invariant checks.
2. **Human-Readable Evaluation Report (`PHASE_12_EVALUATION_REPORT.md`)**:
   - Narrative analysis of the full-sequence results, breakdown of edge cases, discussion of tracking persistence across 525 frames, and dual-space comparison.
3. **Visual Diagnostic Artifacts**:
   - `mot17_09_dual_view_composite.png`: Side-by-side composite panel showing image frame with detections/zones alongside bird's-eye ground map with ground trajectories and ground zones.
   - `mot17_09_ground_heatmap.png`: High-resolution render of the ground plane heatmap.
4. **Conservation & Diagnostic Log (`conservation_audit.log`)**:
   - Line-by-line audit table verifying integer balance across all conservation equations.

---

## 7. Known Scientific & Technical Limitations

1. **Arbitrary Planar Geometry**:
   - Ground coordinates remain strictly `CoordinateFrame.ARBITRARY_PLANAR`. They do **not** represent physical metric units (meters, km/h, $m^2$).
2. **No Real-World Physical Accuracy Claims**:
   - Planar homography accurately captures 2D projective transformation on the calibrated camera surface, but is not validated against surveyed physical ground truth.
3. **Detection & Tracking Evaluation Scope**:
   - Evaluates detection using local greedy IoU matching and tracking using Hungarian bipartite matching (frame-level and sequence-level) for regression and consistency auditing; this is not an official MOTChallenge benchmark or TrackEval submission.
4. **Separation from Short-Window Evidence**:
   - Phase 9 evaluated short windows (10–20 frames) for integration smoke testing. Phase 12 provides full-sequence longevity and stability evidence across 500+ consecutive frames.

---

## 8. Testing Strategy

Phase 12 testing will be structured into three cleanly decoupled test suites:

1. **Pre-Existing Regression Suite (`tests/test_*.py` excluding Phase 12)**:
   - All 201 existing unit and integration tests covering Phases 1–11 must run quickly ($<10$ seconds) and pass 100%.
2. **Phase 12 Unit Tests (`tests/test_evaluation_pipeline.py`)**:
   - Fast unit tests using synthetic/mock sequences to verify evaluation metrics aggregation, report formatting, JSON schema serialization, and invariant audit checks without loading full videos.
3. **Phase 12 Full-Sequence Integration Runner (`tests/test_full_sequence_evaluation.py` or dedicated CLI)**:
   - End-to-end evaluation execution processing full 525 frames of `MOT17-09-FRCNN` and 600 frames of `MOT17-02-FRCNN`.
   - Gated with pytest marker `@pytest.mark.slow` or executed via a dedicated CLI script (`src/people_analytics/evaluation/run_full_evaluation.py`) to prevent slowing down standard test workflows.

---

## 9. API Freeze & Implementation Gate

- **Phases 1–11 API Freeze**: Under no circumstances should Phase 1–11 public classes, methods, or signatures be modified during Phase 12.
- **Implementation Gate**: Phase 12 implementation, test authoring, and full sequence runs **MUST NOT BEGIN** until the USER explicitly reviews and approves this specification document.
