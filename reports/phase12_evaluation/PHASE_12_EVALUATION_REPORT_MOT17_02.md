# Phase 12: Full-System Evaluation Report

## Executive Summary

This evaluation report validates the complete People Movement Analytics pipeline executed across
full MOT17 benchmark sequences under frozen Phase 1–11 contracts. It provides end-to-end evidence
for object detection, multi-object tracking, footpoint extraction, image-space spatial analytics,
homography projection, ground-plane spatial analytics, and cross-space representation comparisons.

> [!IMPORTANT]
> **Scientific Scope & Coordinate Limitation**:
> Ground-plane coordinates operate strictly within `CoordinateFrame.ARBITRARY_PLANAR` using reproducible
> controlled manual calibration fixtures. Arbitrary planar units must **NOT** be interpreted as physical meters,
> nor as evidence of higher physical metric accuracy. Cross-space comparison is strictly a coordinate-representation
> evaluation.

---

## Sequence: `MOT17-02-FRCNN` (600/600 frames @ 30.0 FPS, 1920x1080)

### 1. Detection Evaluation (Image Plane)
- **Protocol**: Phase 2 frozen contract against active pedestrians (`class_id == 1` and `conf == 1.0`).
- **Matching**: Local greedy one-to-one IoU matching (threshold $\ge 0.5$). Non-official MOTChallenge benchmark.
- **True Positives (TP)**: 3667
- **False Positives (FP)**: 445
- **False Negatives (FN)**: 14914
- **Precision**: 0.8918
- **Recall**: 0.1974
- **F1 Score**: 0.3232

### 2. Multi-Object Tracking Evaluation
- **Protocol**: Phase 3 frozen contract against active ground-truth pedestrians.
- **Matching**: Frame-by-frame Hungarian optimal matching for CLEAR MOT / IDSW; sequence-wide Hungarian for IDF1.
- **Total GT Pedestrians**: 18581
- **True Positives (TP)**: 3605 | **FP**: 370 | **FN**: 14976
- **ID Switches (IDSW)**: 52
- **MOTA**: 0.1713
- **IDF1**: 0.2187
- **Tracking Precision**: 0.9069 | **Tracking Recall**: 0.1940

### 3. Trajectory & Motion Dynamics
- **Total Trajectories**: 88
- **Total Accumulated Observations**: 3975
- **Mean Observations per Track**: 45.17
- **Trajectories with Gaps**: 49 (Total Gaps: 121)
- **Gap Duration Statistics (frames)**: Min: 1 | Max: 30 | Mean: 4.55 | Median: 2.0
- **Track Lifespan Statistics (seconds)**: Min: 0.03s | Max: 11.80s | Mean: 1.71s | Median: 0.77s
- **Overall Frame Coverage Ratio**: 0.8783

### 4. Image-Space Analytics Diagnostics
- **Heatmap Accumulated**: 3974 points | **Occupied Cells**: 270 (31.5%) | **Peak Count**: 137
- **Zone Query Records**: 3975 (Inside: 2882, Outside: 1093)
- **Overlapping Zone Breakdown**: Single-zone: 2417 | Multi-zone: 465
- **Dwell Visits**: 234 visits across 79 unique visitors
- **Total Dwell Time**: 103.77s (Average: 0.44s, Max: 8.03s)

### 5. Ground-Space Analytics Diagnostics
- **Coordinate Frame**: `CoordinateFrame.ARBITRARY_PLANAR`
- **Projection State**: Total: 3975 | Valid In-ROI: 2995 | Valid Extrapolated: 980 | Invalid: 0
- **Extrapolation Policy**: `allow_extrapolated=True` (Accepted: 3975, Excluded: 0)
- **Error Code Breakdown**: `{}`
- **Ground Heatmap**: Accumulated: 2995 | Occupied: 295 cells | Peak Count: 116
- **Ground Zone Memberships**: Total: 3975 (Inside: 2539, Outside: 1436, Single: 2065, Multi: 474)
- **Ground Dwell Visits**: 203 visits across 69 unique visitors (Total: 93.67s, Average: 0.46s)

### 6. Cross-Space Representation Comparison
- **Total Visits**: Image: 234 vs Ground: 203
- **Unique Visitors**: Image: 79 vs Ground: 69
- **Total Dwell Duration**: Image: 103.77s vs Ground: 93.67s
- **Observation Agreement Ratio**: 27.50% (1093/3975)

### 7. Mathematical Conservation Audit (AC-03 to AC-07)
| Invariant | Pass/Fail | Equation Balance | Formal Definition |
| :--- | :---: | :--- | :--- |
| AC-03: Projection State Conservation | ✅ PASSED | `3975 == 3975` | N_projected (3975) == In-ROI (2995) + Extrapolated (980) + Invalid (0) |
| AC-04: Image Heatmap Conservation | ✅ PASSED | `3975 == 3975` | N_footpoints (3975) == Accumulated (3974) + OOB (1) + Invalid (0) |
| AC-05: Ground Heatmap Conservation | ✅ PASSED | `3975 == 3975` | N_ground_input (3975) == Accumulated (2995) + OOB (980) + Invalid (0) + ExtrapRejected (0) |
| AC-06a: Image Zone Conservation | ✅ PASSED | `3975 == 3975` | N_footpoints (3975) == Inside (2882) + Outside (1093) |
| AC-06b: Ground Zone Conservation | ✅ PASSED | `3975 == 3975` | N_ground_obs (3975) == Inside (2539) + Outside (1436) |
| AC-07a: Image Dwell Overlap Conservation | ✅ PASSED | `3347 == 3347` | Total Dwell Observations (3347) == Total Zone Assignments (3347) |
| AC-07b: Ground Dwell Overlap Conservation | ✅ PASSED | `3013 == 3013` | Total Ground Dwell Observations (3013) == Total Ground Zone Assignments (3013) |

### 8. Runtime & System Resource Profile (Observational)
- **Total Wall-Clock Time**: 37978.62 ms (37.98 s)
- **Effective Throughput**: **15.80 FPS**
- **Peak Resident Set Size (RSS)**: 439.04 MB
- **Stage Latency Breakdown**:
  - Detection: 37438.76 ms (62.40 ms/frame)
  - Tracking: 166.59 ms (0.28 ms/frame)
  - Footpoint Extraction: 22.59 ms (0.04 ms/frame)
  - Trajectory Building (Dual): 5.99 ms (0.01 ms/frame)
  - Heatmap Accumulation (Dual): 24.32 ms (0.04 ms/frame)
  - Zone Membership (Dual): 94.47 ms (0.16 ms/frame)
  - Dwell Engine (Dual): 13.09 ms (0.02 ms/frame)

---

## Known Limitations & Verification Scope
1. **Arbitrary Planar Space**: Ground coordinates operate in `ARBITRARY_PLANAR` units and do not denote physical meters.
2. **Observational Runtime**: Latency, throughput, and memory metrics are descriptive telemetry only and do not establish pass/fail performance gates.
3. **Local Evaluation Contracts**: Detection and tracking evaluations adhere strictly to frozen custom regression baselines, not official MOTChallenge benchmark submissions.
