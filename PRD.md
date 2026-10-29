# Product Requirements Document (PRD)
## AURI: Computer Vision & Biomedical Engine

---

## 1. Problem Statement
Adhesive capsulitis (frozen shoulder) is an inflammatory and fibrotic condition causing painful restriction of glenohumeral motion. Effective clinical recovery depends heavily on adhering to regular range of motion (ROM) rehabilitation protocols. However:
1. **Clinical Assessment Bottleneck**: In-clinic goniometric assessments occur infrequently, providing sparse progress snapshots.
2. **Home Tracking Unreliability**: Traditional at-home tracking via basic 2D webcam apps suffers from projection distortion, perspective foreshortening, torso compensation (e.g. leaning or shoulder hiking falsely inflating ROM), and measurement jitter.
3. **Black-box AI Failures**: Many computer-vision apps use opaque neural networks that output single confidence numbers or unverified 3D joint positions without physical calibration, coordinate frame definitions, or transparent anatomical interpretation.
4. **Lack of Zero Reference**: Uncalibrated tools report fluctuating initial angles (e.g., oscillating between -3° and +4° at rest), eroding patient trust.

**AURI's Vision**: Provide a modular, deterministic, mathematically transparent, and clinically defensible computer-vision engine that transforms camera observations into verified shoulder kinematics, compensates for trunk movement, and enforces zero-reference neutral zones.

---

## 2. Intended Users & Personas
- **Rehabilitation Researcher**: Requires access to raw and filtered kinematic time series, rigorous coordinate frames, and explicit measurement uncertainty metrics.
- **Physical Therapist / Clinician**: Requires reproducible humerothoracic excursion angles, repetition counts, and alerts for compensatory trunk lean/hiking without unvalidated diagnostic claims.
- **Patient**: Performs prescribed shoulder exercises in front of a standard RGB camera, guided by clear neutral-zone zeroing (resting position is 0°) and stable real-time angular feedback.

---

## 3. Product Goals & Non-Goals

### 3.1 Product Goals
- Transform standard RGB monocular video into robust, explainable shoulder kinematics.
- Decouple raw visual observation (MediaPipe Pose Landmarker) from deterministic biomechanical interpretation.
- Construct a subject-centric orthonormal 3D torso coordinate frame to isolate arm motion from trunk movement.
- Provide a robust baseline calibration procedure yielding a stable 0.0° relative excursion display with hysteresis.
- Quantify compensatory movements (trunk lateral tilt, trunk rotation, trunk flexion/extension, shoulder girdle hiking).
- Output explicit multi-state measurement quality flags (`VALID`, `VALID_WITH_WARNING`, `UNAVAILABLE`, `RECALIBRATION_REQUIRED`).
- Ensure testability through pure mathematical synthetic geometries before camera ingestion.

### 3.2 Non-Goals (Strict Out-of-Scope)
- **NO Autonomous Diagnosis**: The system will not diagnose frozen shoulder or classify clinical stages (e.g., Freezing, Frozen, Thawing).
- **NO Treatment Prescription**: The system will not modify physical therapy regimens autonomously.
- **NO Mobile Frameworks / Apps**: No Flutter, Dart, Android Studio, Kotlin, or iOS code in this repository.
- **NO BLE / Sensor Hardware Integration**: No active IMU Bluetooth streaming in this phase (future interface contracts only).
- **NO Cloud Services or Remote Telemetry**: All core processing is local and privacy-preserving.
- **NO Custom Pose Neural Network Training**: Do not train custom deep learning models; leverage validated MediaPipe observations.

---

## 4. Supported Movements & Movement Protocols

| Movement | Description | Primary Plane | Phase | Status |
|---|---|---|---|---|
| **Forward Shoulder Flexion** | Arm elevation in the sagittal plane ($YZ_T$) | Sagittal | Phase 5 | Confirmed |
| **Shoulder Abduction** | Arm elevation in the frontal/coronal plane ($XZ_T$) | Frontal | Phase 5 | Confirmed |
| **Scaption (Elevation in Scapular Plane)** | Arm elevation ~30°-45° anterior to coronal plane | Scapular | Phase 5 | Confirmed |
| **Humeral Axial Rotation** | Internal / External rotation of the humerus | Transverse | Phase 8 | Proposed / Feasibility Study Required |

> [!IMPORTANT]
> Humeral axial rotation cannot be derived from a single collinear vector connecting shoulder to elbow. Phase 8 will assess camera-only feasibility under specialized protocols (e.g., 90° elbow flexion) or establish necessity of an IMU.

---

## 5. Requirements Catalog

### 5.1 Computer Vision & Pose Ingestion (`CV-REQ`)
| ID | Requirement Statement | Priority | Target Phase | Verification Method |
|---|---|---|---|---|
| `CV-REQ-001` | Ingest video streams from local webcams or video files at $\ge 30$ FPS and $\ge 720\text{p}$. | High | Phase 2 | Integration Test |
| `CV-REQ-002` | Utilize MediaPipe Tasks Vision `PoseLandmarker` for landmark extraction. | High | Phase 2 | Unit / Integration Test |
| `CV-REQ-003` | Preserve both normalized 2D screen coordinates and estimated 3D world landmarks without discarding raw values. | High | Phase 2 | Unit Test |
| `CV-REQ-004` | Enforce landmark presence and visibility thresholds (default $V_{min} = 0.5$) before passing landmarks to kinematics. | High | Phase 2 | Synthetic / Unit Test |
| `CV-REQ-005` | Detect and handle camera view mirroring, left/right laterality, and flipped frames deterministically. | High | Phase 2 | Unit Test |

### 5.2 Mathematical & Biomechanical Modelling (`BIO-REQ`)
| ID | Requirement Statement | Priority | Target Phase | Verification Method |
|---|---|---|---|---|
| `BIO-REQ-001` | Implement safe vector math (dot, cross, norm, unit) with explicit guards against zero-norm and NaN values. | High | Phase 1 | Synthetic / Unit Test |
| `BIO-REQ-002` | Implement unsigned angle between vectors with cosine clamped to $[-1.0, 1.0]$. | High | Phase 1 | Synthetic Test |
| `BIO-REQ-003` | Implement signed angle around a projection normal axis using $\operatorname{atan2}$. | High | Phase 1 | Synthetic Test |
| `BIO-REQ-004` | Derive an orthonormal Torso Frame ($T$) using shoulders ($S_L, S_R$) and hips ($H_L, H_R$) with $+X_T$ Right, $+Y_T$ Anterior, $+Z_T$ Superior. | High | Phase 3 | Synthetic Ground Truth |
| `BIO-REQ-005` | Reject degenerate torso frames when shoulder-hip distance is below numerical threshold or collinear. | High | Phase 3 | Synthetic Test |
| `BIO-REQ-006` | Transform points and direction vectors between World Frame and Torso Frame via verified orthonormal rotation matrices. | High | Phase 3 | Round-trip Unit Test |
| `BIO-REQ-007` | Compute humerothoracic elevation angle relative to downward torso axis $\hat{d}_T = -\hat{z}_T$. | High | Phase 5 | Synthetic Ground Truth |
| `BIO-REQ-008` | Clearly document in API that humerothoracic elevation is total elevation, not isolated glenohumeral ROM. | High | Phase 5 | Documentation / Code Review |

### 5.3 Calibration & Neutral Zone Management (`CAL-REQ`)
| ID | Requirement Statement | Priority | Target Phase | Verification Method |
|---|---|---|---|---|
| `CAL-REQ-001` | Provide a multi-frame baseline calibration mechanism (median of $N=30$ stable frames in resting posture). | High | Phase 6 | Unit / Integration Test |
| `CAL-REQ-002` | Compute relative angular excursion as $\Delta\theta(t) = \theta(t) - \theta_0$. | High | Phase 6 | Unit Test |
| `CAL-REQ-003` | Implement configurable display deadband $\delta$: display $0.0^\circ$ when $|\Delta\theta(t)| \le \delta$. | High | Phase 6 | Unit Test |
| `CAL-REQ-004` | Implement dual-threshold hysteresis ($\delta_{enter} = 3.0^\circ, \delta_{exit} = 5.0^\circ$) to eliminate boundary flicker. | High | Phase 6 | Synthetic Jitter Test |
| `CAL-REQ-005` | Ensure deadband modifies display layer only; raw and filtered excursion values must remain unclipped in state models. | High | Phase 6 | Unit Test |
| `CAL-REQ-006` | Trigger `RECALIBRATION_REQUIRED` if subject relocates, torso frame shifts abruptly, or landmark confidence drops. | Med | Phase 6 | Synthetic Test |

### 5.4 Repetition Detection & Temporal Filtering (`REP-REQ`)
| ID | Requirement Statement | Priority | Target Phase | Verification Method |
|---|---|---|---|---|
| `REP-REQ-001` | Implement deterministic temporal filters (SMA and exponential smoothing) preserving separate raw and filtered signals. | High | Phase 7 | Unit Test |
| `REP-REQ-002` | Implement finite-state machine (FSM) for repetition detection (`IDLE` $\rightarrow$ `RAISING` $\rightarrow$ `PEAK_HOLD` $\rightarrow$ `LOWERING` $\rightarrow$ `COMPLETE`). | High | Phase 7 | Synthetic Time-series Test |
| `REP-REQ-003` | Invalidate repetition progress and reset FSM if tracking quality drops below acceptable threshold during movement. | High | Phase 7 | Synthetic Failure Test |
| `REP-REQ-004` | Record peak excursion angle, time to peak, and repetition duration for every valid repetition. | Med | Phase 7 | Unit Test |

### 5.5 Compensation Detection (`COMP-REQ`)
| ID | Requirement Statement | Priority | Target Phase | Verification Method |
|---|---|---|---|---|
| `COMP-REQ-001` | Measure lateral trunk lean relative to gravity/vertical axis and baseline posture. | High | Phase 7 | Synthetic Test |
| `COMP-REQ-002` | Measure trunk axial rotation relative to camera viewing axis. | Med | Phase 7 | Synthetic Test |
| `COMP-REQ-003` | Detect shoulder hiking by tracking clavicular/acromial elevation relative to the sternal/torso frame. | High | Phase 7 | Synthetic Test |
| `COMP-REQ-004` | Emit discrete compensation warnings (`LEAN_DETECTED`, `HIKING_DETECTED`) when thresholds are exceeded. | High | Phase 7 | Unit Test |

### 5.7 Movement-Specific Analysis (`MOV-REQ`)

| ID | Requirement | Priority | Phase | Verification |
|---|---|---|---|---|
| `MOV-REQ-001` | Define and compute a movement-specific angle for flexion, extension, abduction, adduction, scaption, horizontal abduction/adduction, with documented reference, sign, range and validity region. | High | Post-7 | Synthetic Test |
| `MOV-REQ-002` | Axial rotation (internal/external) only under the documented elbow-90 protocol, always flagged low confidence; otherwise NOT_MEASURABLE. | High | Post-7 | Synthetic Test |
| `MOV-REQ-003` | Movement selection is explicit; movement suggestion is advisory, separate and reports ambiguity. | High | Post-7 | Unit Test |
| `MOV-REQ-004` | Pose outside a movement's definition, low confidence or poor view never yields a misleading number (angle None / flagged). | High | Post-7 | Synthetic Failure Test |
| `MOV-REQ-005` | Motion estimates carry confidence, status, view and reasons; the calibrated channel and repetition FSM track the selected movement's metric. | High | Post-7 | Integration Test |
| `MOV-REQ-006` | Optional circumduction reported as a compound movement (sweep + components), not a single angle. | Low | Post-7 | Synthetic Test |

### 5.6 Measurement Quality & System Health (`QUAL-REQ`)
| ID | Requirement Statement | Priority | Target Phase | Verification Method |
|---|---|---|---|---|
| `QUAL-REQ-001` | Maintain explicit quality status enum: `VALID`, `VALID_WITH_WARNING`, `UNAVAILABLE`, `RECALIBRATION_REQUIRED`. | High | Phase 2 | Unit Test |
| `QUAL-REQ-002` | Never output $0.0^\circ$ or valid angles when landmarks are missing or quality is `UNAVAILABLE`. | High | Phase 6 | Unit Test |
| `QUAL-REQ-003` | Quantify pose jitter as the rolling standard deviation of landmark positions over a temporal window. | Med | Phase 7 | Synthetic Test |

### 5.7 Privacy & Data Handling (`PRIV-REQ`)
| ID | Requirement Statement | Priority | Target Phase | Verification Method |
|---|---|---|---|---|
| `PRIV-REQ-001` | Do not persist raw video or audio to disk by default. | High | Phase 2 | Architecture Review |
| `PRIV-REQ-002` | Store only derived numerical kinematics and quality telemetry in session artifacts. | High | Phase 2 | Architecture Review |

---

## 6. Non-Functional Requirements
- **Performance**: Pipeline latency $\le 50\,\text{ms}$ per frame on standard modern CPU (x86_64, Apple Silicon).
- **Determinism**: 100% reproducible kinematic output given the same sequence of landmark coordinates.
- **Modularity**: Zero dependency of the geometry and kinematics engine on OpenCV, UI, or camera hardware.
- **Maintainability**: Fully typed codebase conforming to PEP 8 / Ruff, with comprehensive docstrings explaining the mathematical derivations.

---

## 7. Dependencies, Risks & Open Questions
- **Risk 1 (Monocular Depth Uncertainty)**: MediaPipe's 3D world landmarks use a synthetic heuristic model trained on monocular imagery; depth ($Z$) is less accurate than in-plane ($X, Y$).
  - *Mitigation*: Prioritize planar projections aligned with camera view (e.g., frontal view for abduction, sagittal view for flexion) and document limitations.
- **Risk 2 (Axial Rotation Feasibility)**: Surface landmarks from shoulder to elbow form a 1D line segment, offering zero longitudinal rotational observability.
  - *Mitigation*: Conduct feasibility study in Phase 8; design explicit contract for upper-arm IMU integration.
- **Open Question 1**: What are the optimal clinician-accepted deadband thresholds ($\delta_{enter}, \delta_{exit}$) for shoulder elevation across various patient cohorts?
  - *Current Status*: Initial engineering values set to $3.0^\circ$ and $5.0^\circ$; subject to validation against physical goniometer.

