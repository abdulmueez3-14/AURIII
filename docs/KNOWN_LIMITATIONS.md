# Known Limitations and Observability Boundaries
## AURI: Computer Vision & Biomedical Engine

---

## 1. Observability and Monocular Vision Limitations

### 1.1 Inherent Depth ($Z$) Uncertainty
- **Limitation**: Standard RGB cameras lack time-of-flight or stereoscopic depth sensors.
- **Consequence**: MediaPipe's 3D "world landmarks" rely on statistical depth priors inferred from monocular 2D images. Depth estimates ($Z$) exhibit substantially higher noise and variance than in-plane ($X, Y$) coordinates.
- **Mitigation in AURI**: Kinematic protocols are structured to encourage planar motion aligned with the camera's optical axis (e.g., frontal view for abduction, 90° lateral sagittal view for forward flexion).

### 1.2 Unobservability of Humeral Axial Rotation
- **Limitation**: The landmarks defining the upper arm are the shoulder joint center $\mathbf{S}$ and the elbow joint center $\mathbf{E}$. These two points define a 1D line segment in $\mathbb{R}^3$.
- **Consequence**: Rotation of the humerus around its own longitudinal axis (internal/external rotation) produces zero displacement of the shoulder and elbow joint centers.
- **Geometric Reality**: It is mathematically impossible to observe humeral axial rotation from the shoulder-elbow segment alone.
- **Protocol Requirements**: Observing axial rotation via camera requires the elbow to be flexed to 90°, using the forearm as an indicator arm. However, even with 90° elbow flexion, monocular foreshortening creates severe ambiguity. Phase 8 will conduct a formal feasibility study, and an upper-arm IMU sensor will likely be required for clinical-grade internal/external rotation tracking.

### 1.3 Perspective Foreshortening in Extreme Elevation
- When the arm is elevated directly toward or away from the camera lens, landmark spacing compresses in the 2D image plane, leading to model localization jitter.

---

## 2. Biomechanical and Anatomical Boundaries

### 2.1 Humerothoracic vs. Glenohumeral Motion (Scapular Motion)
- **Limitation**: Surface computer vision cannot observe the underlying scapula beneath muscle and skin.
- **Consequence**: The angle measured between the torso and the humerus represents total **humerothoracic elevation**. During abduction, roughly $2/3$ of the elevation occurs at the glenohumeral joint and $1/3$ at the scapulothoracic articulation.
- **Boundary**: AURI does not claim isolated glenohumeral range of motion.

### 2.2 Functional Starting Posture vs. True Anatomical Zero
- In adhesive capsulitis, resting shoulder posture is often pathologically altered (capsular contracture).
- Zeroing relative to the patient's resting baseline ($\Delta\theta = 0^\circ$) tracks functional range of motion during exercise, but does not indicate that the patient achieved true anatomical neutral.

---

## 3. Environmental and Subject Factors

| Factor | Manifestation | Engine Response |
|---|---|---|
| **Loose or Baggy Clothing** | Shifts visual joint centers away from true skeletal landmarks. | Emits warning on high landmark jitter; clinician setup guide recommends form-fitting attire. |
| **Partial Occlusion** | Arm passes behind head or torso, or exits camera framing. | Landmark visibility drops $< 0.5$; engine immediately outputs `UNAVAILABLE`. |
| **Rapid Movement / Motion Blur** | Landmark localization failure due to shutter blur. | Temporal velocity checks flag abnormal acceleration; drops invalid frames. |
| **Subject Relocation** | Patient steps away from calibrated position or shifts chair. | Discontinuity in torso frame position $> 0.15\,\text{m}$ invalidates baseline; triggers `RECALIBRATION_REQUIRED`. |
| **Low Ambient Light** | Sensor noise corrupts feature extraction. | Pose estimator returns low presence scores; rejected at quality gate. |

---

## 4. Clinical Safety Boundaries
- **No Autonomous Staging**: AURI will not assign frozen shoulder clinical staging (Freezing, Frozen, Thawing).
- **No Direct Medical Advice**: Feedback is restricted to objective numerical measurements (degrees of excursion, repetition counts, trunk lean warnings).
- **Clinician Review Required**: All exercise thresholds, rep targets, and compensation limits must be configured and approved by licensed healthcare professionals.


## Calibration: Relocation and Torso-Shift Detection (Phase 6)
- `CAL-REQ-006` is only partially implemented. Recalibration is requested from the quality gate and from tracking gaps longer than 15 frames. A subject who steps sideways, turns, or moves toward the camera while tracking stays confident is NOT detected, because MediaPipe world landmarks are hip-centred and carry no absolute translation. Planned: monitor torso orientation and shoulder-width scale against the calibration reference.
- Deadband (3/5 deg) and acceptance gate (IQR 3 deg) values are engineering defaults without clinical validation.

## Shoulders-Only Mode (ADR-010)
- Vertical is assumed, not measured: camera tilt and trunk lean are folded into the angles. No lean/hiking metrics. The Y-down vertical assumption for MediaPipe world coordinates has not been validated on real data.

## Movement-Specific Analysis (ADR-011)
- See docs/MOVEMENT_ANALYSIS.md section 7. Highlights: horizontal abduction/adduction and rotation are depth-sensitive and low confidence from a single camera; confidence factors are uncalibrated heuristics; camera yaw cannot tell front from back; synthetic tests prove the geometry, not clinical accuracy; no IMU, dashboard or simulator exists in this repository.
