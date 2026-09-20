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

## Sequence: `MOT17-09-FRCNN` (525/525 frames @ 30.0 FPS, 1920x1080)

### 1. Detection Evaluation (Image Plane)
- **Protocol**: Phase 2 frozen contract against active pedestrians (`class_id == 1` and `conf == 1.0`).
- **Matching**: Local greedy one-to-one IoU matching (threshold $\ge 0.5$). Non-official MOTChallenge benchmark.
- **True Positives (TP)**: 3456
- **False Positives (FP)**: 659
- **False Negatives (FN)**: 1869
- **Precision**: 0.8399
- **Recall**: 0.6490
- **F1 Score**: 0.7322

### 2. Multi-Object Tracking Evaluation
- **Protocol**: Phase 3 frozen contract against active ground-truth pedestrians.
- **Matching**: Frame-by-frame Hungarian optimal matching for CLEAR MOT / IDSW; sequence-wide Hungarian for IDF1.
- **Total GT Pedestrians**: 5325
- **True Positives (TP)**: 3399 | **FP**: 590 | **FN**: 1926
- **ID Switches (IDSW)**: 95
- **MOTA**: 0.5097
- **IDF1**: 0.4872
- **Tracking Precision**: 0.8521 | **Tracking Recall**: 0.6383

### 3. Trajectory & Motion Dynamics
- **Total Trajectories**: 85
- **Total Accumulated Observations**: 3989
- **Mean Observations per Track**: 46.93
- **Trajectories with Gaps**: 43 (Total Gaps: 115)
- **Gap Duration Statistics (frames)**: Min: 1 | Max: 30 | Mean: 6.98 | Median: 3.0
- **Track Lifespan Statistics (seconds)**: Min: 0.03s | Max: 17.47s | Mean: 1.88s | Median: 1.00s
- **Overall Frame Coverage Ratio**: 0.8324

### 4. Image-Space Analytics Diagnostics
- **Heatmap Accumulated**: 3986 points | **Occupied Cells**: 235 (27.4%) | **Peak Count**: 361
- **Zone Query Records**: 3989 (Inside: 2370, Outside: 1619)
- **Overlapping Zone Breakdown**: Single-zone: 2074 | Multi-zone: 296
- **Dwell Visits**: 128 visits across 54 unique visitors
- **Total Dwell Time**: 84.60s (Average: 0.66s, Max: 4.87s)

### 5. Ground-Space Analytics Diagnostics
- **Coordinate Frame**: `CoordinateFrame.ARBITRARY_PLANAR`
- **Projection State**: Total: 3989 | Valid In-ROI: 2144 | Valid Extrapolated: 1845 | Invalid: 0
- **Extrapolation Policy**: `allow_extrapolated=True` (Accepted: 3989, Excluded: 0)
- **Error Code Breakdown**: `{}`
- **Ground Heatmap**: Accumulated: 2144 | Occupied: 242 cells | Peak Count: 98
- **Ground Zone Memberships**: Total: 3989 (Inside: 1663, Outside: 2326, Single: 1443, Multi: 220)
- **Ground Dwell Visits**: 85 visits across 42 unique visitors (Total: 59.93s, Average: 0.71s)

### 6. Cross-Space Representation Comparison
- **Total Visits**: Image: 128 vs Ground: 85
- **Unique Visitors**: Image: 54 vs Ground: 42
- **Total Dwell Duration**: Image: 84.60s vs Ground: 59.93s
- **Observation Agreement Ratio**: 40.59% (1619/3989)

### 7. Mathematical Conservation Audit (AC-03 to AC-07)
| Invariant | Pass/Fail | Equation Balance | Formal Definition |
| :--- | :---: | :--- | :--- |
| AC-03: Projection State Conservation | ✅ PASSED | `3989 == 3989` | N_projected (3989) == In-ROI (2144) + Extrapolated (1845) + Invalid (0) |
| AC-04: Image Heatmap Conservation | ✅ PASSED | `3989 == 3989` | N_footpoints (3989) == Accumulated (3986) + OOB (3) + Invalid (0) |
| AC-05: Ground Heatmap Conservation | ✅ PASSED | `3989 == 3989` | N_ground_input (3989) == Accumulated (2144) + OOB (1845) + Invalid (0) + ExtrapRejected (0) |
| AC-06a: Image Zone Conservation | ✅ PASSED | `3989 == 3989` | N_footpoints (3989) == Inside (2370) + Outside (1619) |
| AC-06b: Ground Zone Conservation | ✅ PASSED | `3989 == 3989` | N_ground_obs (3989) == Inside (1663) + Outside (2326) |
| AC-07a: Image Dwell Overlap Conservation | ✅ PASSED | `2666 == 2666` | Total Dwell Observations (2666) == Total Zone Assignments (2666) |
| AC-07b: Ground Dwell Overlap Conservation | ✅ PASSED | `1883 == 1883` | Total Ground Dwell Observations (1883) == Total Ground Zone Assignments (1883) |

### 8. Runtime & System Resource Profile (Observational)
- **Total Wall-Clock Time**: 35034.98 ms (35.03 s)
- **Effective Throughput**: **14.99 FPS**
- **Peak Resident Set Size (RSS)**: 433.36 MB
- **Stage Latency Breakdown**:
  - Detection: 34511.35 ms (65.74 ms/frame)
  - Tracking: 177.13 ms (0.34 ms/frame)
  - Footpoint Extraction: 21.94 ms (0.04 ms/frame)
  - Trajectory Building (Dual): 5.47 ms (0.01 ms/frame)
  - Heatmap Accumulation (Dual): 21.69 ms (0.04 ms/frame)
  - Zone Membership (Dual): 94.60 ms (0.18 ms/frame)
  - Dwell Engine (Dual): 13.09 ms (0.02 ms/frame)

---

## Known Limitations & Verification Scope
1. **Arbitrary Planar Space**: Ground coordinates operate in `ARBITRARY_PLANAR` units and do not denote physical meters.
2. **Observational Runtime**: Latency, throughput, and memory metrics are descriptive telemetry only and do not establish pass/fail performance gates.
3. **Local Evaluation Contracts**: Detection and tracking evaluations adhere strictly to frozen custom regression baselines, not official MOTChallenge benchmark submissions.
