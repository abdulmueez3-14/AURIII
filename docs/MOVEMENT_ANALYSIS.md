# MOVEMENT_ANALYSIS.md — Movement-Specific Shoulder Measurement

Implements `MOV-REQ-001` to `MOV-REQ-006` (PRD.md §5.7). Code: `src/auri_cv/movements/shoulder_movements.py`,
`circumduction.py`; pipeline wiring in `src/auri_cv/pipeline.py`. Tests:
`tests/synthetic/test_shoulder_movements.py`, `tests/synthetic/test_pipeline_movements.py`.

**Status: engineering implementation verified on synthetic geometry only. Nothing here is clinically
validated; every number needs comparison with a reference measurement system before clinical use.**

## 1. Root cause of the earlier "single metric" behaviour

The engine never computed a rotation. Its tracked channel was **total humerothoracic elevation**
(angle from the downward torso axis), which is plane-independent by construction. Flexion, abduction and
plane of elevation existed only as raw projection angles. There was **no movement layer**: no
definition of which angle belongs to which exercise, no extension/adduction/horizontal/scaption semantics,
no validity regions, no per-movement confidence, and axial rotation was deliberately rejected (ADR-006).
Because every exercise was tracked through the same elevation number, flexion, abduction and scaption were
indistinguishable and rotation was absent.

## 2. Architecture

```
ArmSegmentState (torso frame)
   ├── estimate_motion(movement, segment, quality, camera_yaw)   → MotionEstimate  (selected movement)
   ├── suggest_movement(segment)                                 → MovementSuggestion (hint only)
   └── CircumductionTracker (compound)                           → sweep + components
```
* **Selection is explicit** (`ShoulderAnalysisPipeline(movement=...)`, `run_camera.py --movement`).
  Classification is separate and advisory: several poses are intrinsically ambiguous (arm forward at 90° is
  both flexion and horizontal adduction; at 90° elevation scaption plane overlaps horizontal adduction), so
  `suggest_movement` returns the *candidate set* and sets `movement` only when exactly one definition fits.
  Rotation is never inferred by the suggester.
* The calibrated channel (baseline, deadband, repetition FSM) tracks **the selected movement's own angle**.
  With no movement selected the legacy elevation behaviour is unchanged.
* A pose outside a movement's definition gives `NOT_MEASURABLE` (angle `None`) — never a borrowed number —
  and is treated as *pose-gated*, not as tracking loss (it does not latch recalibration).

## 3. Conventions

Torso frame T: +X right, +Y anterior, +Z superior. `u` = shoulder→elbow unit vector in T.
`x_lat = +u_x` (right arm) or `-u_x` (left arm), so “lateral” is positive for both sides.
`elev = arccos(-u_z)` ∈ [0°,180°]; plane of elevation `φ = atan2(u_y, x_lat)` (0° lateral, +90° anterior).
Output angles are in degrees, positive in the movement's own direction; a negative value means the arm is in
the antagonist direction (flagged in `reasons`, `within_nominal_range=False`).

## 4. Movement definitions

| Movement | Reference (0°) | Calculation | + direction | Nominal range | Validity region (else NOT_MEASURABLE) | Preferred view |
|---|---|---|---|---|---|---|
| Flexion | arm hanging (−Z_T) | `atan2(u_y, −u_z)` | forward | 0–180° | arm ≤25° out of the sagittal plane | sagittal (side-on) |
| Extension | arm hanging | `−atan2(u_y, −u_z)` | backward | 0–60° | same | sagittal |
| Abduction | arm hanging | `atan2(x_lat, −u_z)` | away from midline | 0–180° | arm ≤25° out of the coronal plane | frontal |
| Adduction | arm hanging | `−atan2(x_lat, −u_z)` | across midline | 0–30° | same | frontal |
| Scaption | arm hanging | `elev` | elevation | 0–180° | plane of elevation within ±20° of 30° (checked when elev ≥ 20°) | oblique / frontal |
| Horizontal adduction | arm abducted 90° (lateral) | `atan2(u_y, x_lat)` | toward/across the front | 0–130° | `|elev − 90°| ≤ 25°` | frontal (low depth confidence) |
| Horizontal abduction | arm abducted 90° | `−atan2(u_y, x_lat)` | backward | 0–45° | same | frontal (low depth confidence) |
| External rotation | forearm anterior, elbow ≈90°, arm at side | signed angle about `u` from anterior to the forearm component ⟂ `u` (sign flipped for the left arm) | forearm swings outward | 0–90° | elbow flexion 70–110°, `elev ≤ 30°`, wrist visible, forearm ⟂ component ≥ 0.3 | frontal |
| Internal rotation | same | negative of the external-rotation angle | forearm swings inward | 0–70° | same | frontal |
| Circumduction (optional, compound) | — | unwrapped plane-of-elevation sweep ≥ 300° while `elev ≥ 30°`; components = sectors traversed | — | — | tracking loss or low elevation restarts | any |

Elbow vs shoulder: shoulder angles use only the shoulder→elbow vector. The forearm is used **only** for rotation.
A test verifies that bending the elbow with an unchanged upper arm does not change a shoulder angle.

## 5. Confidence (heuristic, not an accuracy bound)

`confidence = min(cap, quality_score × view_factor × plane_factor)`
* `quality_score` – mean landmark visibility from the quality gate.
* `view_factor` – 1.0 preferred view, 0.7 acceptable, 0.4 poor, 0.6 unknown. The view is estimated per frame
  from the shoulder-line yaw relative to the image plane (0° frontal, 90° side-on; front/back and left/right
  are indistinguishable). Assumes camera-aligned world axes (true for MediaPipe world landmarks).
* `plane_factor` – falls from 1.0 to 0.5 as the arm approaches the edge of the movement's validity region.
* `cap` – ceiling for depth-sensitive movements: horizontal 0.6, rotation 0.5, scaption 0.85, others 0.9.

Status: `OK` ≥ 0.6; `LOW_CONFIDENCE` 0.25–0.6 (angle returned, must be labelled); `UNRELIABLE` < 0.25 (angle
withheld); `NOT_MEASURABLE` (angle `None`). All constants are engineering defaults, **not calibrated**.

## 6. What the available setup can and cannot measure

| Movement | Camera only (this repo) | What would improve / is missing |
|---|---|---|
| Flexion, extension | Estimable from a side view; weak from the front (forward motion is monocular depth) | Side camera or depth sensor; waist IMU for gravity-referenced trunk |
| Abduction, adduction | Estimable from a front/back view | Waist IMU to separate trunk lean from arm motion |
| Scaption | Estimable (oblique/frontal), plane tolerance enforced | As above |
| Horizontal abduction/adduction | **Low reliability**: transverse-plane motion is mostly depth from the front | Overhead/top camera or a second camera; upper-arm IMU |
| External/internal rotation | **Experimental, protocol-only** (elbow 90°, arm at side, wrist visible). Monocular depth makes the forearm's forward/back component weak | Upper-arm IMU (humeral orientation), or a second camera; an IMU pair only works with the same protocol and is limited by magnetometer-free heading drift |
| Circumduction | Compound detector only; no single angle | — |

IMU notes (contract only; IMU code is out of scope, see ARCHITECTURE §5): wrist and waist IMUs alone **cannot**
establish every shoulder angle. A wrist IMU measures forearm orientation, which mixes shoulder, elbow and
forearm motion; a waist IMU measures trunk orientation. Gravity from the waist IMU would replace the assumed
vertical in shoulders-only mode and resolve trunk lean. Axial humeral rotation at arbitrary poses needs an
upper-arm sensor or an additional camera view. These requirements are stated, not worked around.

## 7. Known limitations

* No clinical validation; the 25°/20° tolerances, confidence factors and caps are not empirically derived.
* Camera views: yaw cannot distinguish front from back or left from right side-on.
* Rotation uses the monocular forearm direction; accuracy is unknown and flagged `LOW_CONFIDENCE`.
* Shoulders-only mode folds trunk lean into the angles (ADR-010); movement angles are then relative to the
  assumed vertical rather than the trunk.
* The suggestion layer is ambiguous by design for several poses; it never overrides the selection.
