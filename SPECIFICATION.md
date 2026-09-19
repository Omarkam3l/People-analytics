# System Specification

## 1. Project Definition

People Movement Analytics is a Computer Vision pipeline that transforms fixed-camera video into structured information about where people move and how long they remain in specific areas.

The system operates on individual video streams in the initial version.

---

# 2. Functional Requirements

## FR-01 — Video Input

The system must accept a video stream or video file containing people.

The initial implementation targets fixed-camera video.

---

## FR-02 — Person Detection

The system must detect people in each processed frame.

Each detection must contain at minimum:

- bounding box
- confidence score
- class label

Only the `person` class is required for the initial system.

---

## FR-03 — Multi-Object Tracking

The system must associate detections across frames.

Each tracked person must receive a stable `track_id`.

The tracker must maintain the identity of a person across consecutive frames whenever possible.

---

## FR-04 — Footpoint Extraction

For each tracked person's bounding box, the system must calculate a footpoint.

Initial definition:

The footpoint is the bottom-center point of the bounding box.

For bounding box:

(x1, y1, x2, y2)

the footpoint is:

x = (x1 + x2) / 2
y = y2

The implementation must keep the footpoint calculation isolated from detection and tracking logic.

---

## FR-05 — Trajectory Generation

The system must maintain a sequence of footpoints for each `track_id`.

Example:

track_id = 17

t0 → (x0, y0)
t1 → (x1, y1)
t2 → (x2, y2)
...

A trajectory must preserve temporal ordering.

---

## FR-06 — Heatmap

The system must generate a spatial heatmap based on footpoint positions.

The initial heatmap operates in image coordinates.

Each valid footpoint contributes to the spatial density representation.

The implementation must support configurable spatial resolution.

---

## FR-07 — Zones

The system must support user-defined spatial zones.

A zone represents a region of interest in the camera view.

The initial implementation may use polygonal zones.

---

## FR-08 — Zone Membership

For every valid footpoint, the system must determine whether the point lies inside a defined zone.

Zone membership must be evaluated using the footpoint rather than the center of the bounding box.

---

## FR-09 — Dwell Time

The system must calculate the amount of time a tracked person remains inside a zone.

Dwell time must be derived from timestamps or frame indices and the video's FPS.

For a person entering a zone at time:

t_enter

and leaving at:

t_exit

dwell time is:

dwell_time = t_exit - t_enter

---

## FR-10 — Multiple Visits

The system must support multiple visits to the same zone by the same person.

Example:

Person 17:

Zone A
Visit 1 → 12 seconds
Visit 2 → 8 seconds

Total dwell time:

20 seconds

The system must not incorrectly merge separated visits.

---

## FR-11 — Zone Analytics

The system must provide aggregate statistics per zone.

At minimum:

- number of unique visitors
- number of visits
- total dwell time
- average dwell time
- maximum dwell time

---

## FR-12 — Visualization

The system must provide visualization for:

- bounding boxes
- track IDs
- footpoints
- trajectories
- zones
- heatmaps

The visualization layer must remain separate from the analytics logic.

---

# 3. Data Model

A tracked observation should conceptually contain:

```text
timestamp
frame_index
track_id
bounding_box
confidence
footpoint
zone_id