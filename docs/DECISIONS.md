# Architecture Decision Records (ADRs)
## AURI: Computer Vision & Biomedical Engine

---

## ADR-001: Standalone Python Architecture with Zero Legacy Coupling
- **Status**: Accepted
- **Context**: The project workspace sits in a directory tree with legacy prototype scripts, PowerPoint summaries, and SRS documents (`..\build`). Previous iterations mixed Flutter/Dart mobile shells with exploratory CV scripts.
- **Decision**: Establish AURI CV as a completely standalone, self-contained Python project in `c:\Users\abdul\Topics\medic 2026\AURI`. Zero lines of code are copied from legacy trackers. No mobile, Flutter, Android, or cloud dependencies will be installed.
- **Rationale**: Clean separation enables mathematical rigor, rapid test cycles, and high code quality without legacy technical debt.
- **Consequences**: Future mobile integration will communicate via documented data schemas (JSON / Protobuf) rather than shared code.
- **Revisit Conditions**: When the Python biomedical engine is fully validated and a mobile deployment packaging phase is formally commissioned.

---

## ADR-002: Adoption of Python 3.12 LTS and MediaPipe Tasks Vision API
- **Status**: Accepted
- **Context**: Python 3.14 was present on the host system, but experimental pre-releases often lack prebuilt C-extension wheels. Furthermore, MediaPipe v1.2.0 has permanently removed legacy `mp.solutions.pose` in favor of `mediapipe.tasks.python.vision.PoseLandmarker`.
- **Decision**: Pin to CPython 3.12.10 via `uv` virtual environment and use the modern MediaPipe Tasks Vision `PoseLandmarker` API.
- **Rationale**: Guarantees rock-solid wheel availability across Windows x86_64 for NumPy 2.x, SciPy, OpenCV, and MediaPipe. Aligns with Google's official MediaPipe roadmap.
- **Consequences**: Must use `.task` bundle model assets and Tasks Vision configuration options instead of legacy solutions code.
- **Revisit Conditions**: When Python 3.14 wheels become officially standard and supported across all dependencies.

---

## ADR-003: Subject-Centered Right-Handed Torso Frame ($+X$ Right, $+Y$ Anterior, $+Z$ Superior)
- **Status**: Accepted
- **Context**: Video cameras provide camera-relative coordinates. If a subject leans, tilts, or turns slightly, camera-relative angles fluctuate wildly and confound arm elevation with trunk posture.
- **Decision**: Establish an anatomical right-handed Torso Coordinate Frame ($T$) using shoulders ($S_L, S_R$) and hips ($H_L, H_R$):
  - $+X_T$: Subject's anatomical right
  - $+Y_T$: Anterior (ventral, projecting forward from chest)
  - $+Z_T$: Superior (headward along spine axis)
- **Rationale**: Isolates humerothoracic arm movement from trunk movements. Orthonormal basis guarantees safe Euclidean transformations without scaling distortions.
- **Consequences**: Requires Gram-Schmidt orthogonalization on every frame and rejection of degenerate/collinear torso landmark inputs.
- **Revisit Conditions**: If clinical research explicitly requests a specialized ISB (International Society of Biomechanics) thorax frame requiring sternum/vertebral markers not observable on RGB cameras.

---

## ADR-004: Decoupling Geometric Observation from Biomechanical Interpretation
- **Status**: Accepted
- **Context**: Monocular pose estimators output joint landmark coordinates, but possess no intrinsic concept of joint constraints, calibration offsets, or clinical definitions.
- **Decision**: Strict two-tier architecture:
  1. Observation Tier: MediaPipe Tasks generates raw 2D and 3D coordinate observations.
  2. Interpretation Tier: AURI's deterministic geometry engine handles normalization, coordinate frames, kinematics, neutral zeroing, and quality scoring.
- **Rationale**: Testability. Mathematical modules can be verified with 100% determinism against synthetic geometric inputs without running video or neural networks.
- **Consequences**: Zero neural network training required in AURI; all biomechanical logic is open, explainable, and inspectable.
- **Revisit Conditions**: Only if empirical reference comparisons demonstrate systematic pose landmark failures that cannot be corrected geometrically.

---

## ADR-005: Mandatory Baseline Calibration and Display Deadband with Raw Data Preservation
- **Status**: Accepted
- **Context**: In clinical trials and home exercises, resting postures exhibit subtle physiological tremor and camera landmark jitter ($\pm 1^\circ$ to $\pm 3^\circ$). Continuous display flickering around zero destroys user confidence.
- **Decision**: Implement a mandatory baseline calibration step ($\theta_0 = \text{median of } N \text{ frames}$) and a dual-threshold display deadband ($\delta_{\text{enter}} = 3.0^\circ, \delta_{\text{exit}} = 5.0^\circ$).
  - Display angle is forced to $0.0^\circ$ while in the neutral zone.
  - Raw and filtered excursion angles are preserved without clipping in data packets.
- **Rationale**: Provides rock-solid UI stability while preserving uncorrupted scientific data for researchers and clinician review.
- **Consequences**: Display layer is strictly separated from analytical data layer.
- **Revisit Conditions**: If clinical feedback suggests tighter or wider deadband thresholds for specific pathologies.

---

## ADR-006: Rejection of Single-Vector Humeral Axial Rotation Calculation
- **Status**: Accepted
- **Context**: Users and legacy apps often attempt to compute shoulder internal/external rotation from frontal video landmarks.
- **Decision**: Formally reject computing humeral axial rotation from the shoulder-to-elbow vector. Mark axial rotation as unobservable via simple vector kinematics.
- **Rationale**: The shoulder-to-elbow vector is a 1D line segment. Rotation around its own axis produces zero landmark displacement. Claiming to measure axial rotation without 90° elbow flexion or an upper-arm IMU is mathematically invalid and clinically unsafe.
- **Consequences**: Phase 8 is dedicated to a formal feasibility study; internal/external rotation will be designated as requiring either a strict 90° elbow protocol or an IMU sensor.
- **Revisit Conditions**: Completion of Phase 8 feasibility study.

---

## ADR-007: Prioritization of Synthetic Geometric Ground-Truth Tests Before Camera Ingestion
- **Status**: Accepted
- **Context**: Testing kinematics solely on live webcam feeds introduces uncontrolled variability (lighting, clothing, human tremor), making it impossible to distinguish algorithmic bugs from pose detection errors.
- **Decision**: Build an automated synthetic test suite in `tests/synthetic/` with analytically known 3D point geometries (0°, 30°, 45°, 90°, 120°, 180°, perturbations, and noise) before connecting camera feeds.
- **Rationale**: Ensures mathematical code is provably correct, numerically robust, and regression-tested with zero external dependencies.
- **Consequences**: High upfront test development; near-zero downstream kinematics bugs.
- **Revisit Conditions**: Continuous; standard practice throughout all phases.


---

## ADR-008: Planar Projection Angles Are Undefined (None) When the Arm Leaves the Plane
- **Status**: Accepted
- **Context**: Sagittal flexion `atan2(v_y, -v_z)`, coronal abduction `atan2(x_lat, -v_z)` and plane of elevation `atan2(v_y, x_lat)` evaluate `atan2` on the in-plane component. When that component is ~0 (e.g. flexion for a purely lateral arm, plane of elevation for a vertical arm) the result is rounding noise (flipping between 0 and 180 deg) or a fabricated 0.
- **Decision**: Each solver raises `DegenerateVectorError` when the in-plane component is below `sin(5 deg)`; `solve_arm_kinematics` maps this to `None` fields in `RawKinematics`. Total humerothoracic elevation is always defined and is never gated.
- **Rationale**: Consistent with the "Failure Transparency" rule: never present an unobservable angle as a number. Supersedes the VALIDATION_PLAN expectation of flexion 0 deg for pure abduction.
- **Consequences**: Downstream calibration/display code must handle `None` per-angle. The 5 deg threshold is a tunable engineering default, not clinically validated.
- **Revisit Conditions**: Phase 6/7 choice of primary tracked angle per exercise.

---

## ADR-009: Compensation Measured Relative to Baseline Torso Frame; Repetitions Require Arming
- **Status**: Accepted
- **Context**: MATHEMATICAL_MODEL 10.1 measures lean against a world vertical, but the world frame convention differs between fixtures (Z-up) and MediaPipe world landmarks (Y-down), and the camera may be tilted.
- **Decision**: Trunk lateral/forward lean and rotation are the components of the current trunk axes expressed in the baseline (calibration) TorsoFrame. Hiking uses the 10.2 formula unchanged. The repetition FSM only starts a repetition after the signal has been in the resting zone (<= return threshold).
- **Rationale**: Convention-independent and camera-tilt-invariant (tested); reports change since the clinician-approved starting posture. Arming prevents counting partial movements seen after session start or tracking gaps.
- **Consequences**: Not an absolute gravity-referenced lean: a subject who calibrated while leaning will show 0. A baseline TorsoFrame must be stored at calibration.
- **Revisit Conditions**: If gravity-referenced lean is required (e.g. IMU), or camera tilt can be estimated.

---

## ADR-010: Opt-in Shoulders-Only Torso Frame (Assumed Vertical)
- **Status**: Accepted (opt-in; default remains hips-based)
- **Context**: First live session (Iteration 7.3): hips were below visibility threshold for most frames (seated / close to camera), so no angle was ever produced. Rehabilitation use is often seated.
- **Decision**: `TorsoMode.SHOULDERS_ONLY_ASSUMED_VERTICAL` builds the torso frame from the two shoulders and an ASSUMED world vertical (`MEDIAPIPE_WORLD_UP = (0,-1,0)`); the quality gate then does not require hips. Frame carries `hips_observed=False`. The mode is fixed per session so baseline and live frames share a reference. Lean, rotation and hiking are disabled in this mode.
- **Rationale**: Makes measurement possible when hips are not visible without fabricating hip data. The default mode is unchanged.
- **Consequences**: Elevation becomes relative to the world vertical, not the trunk, so a leaning trunk is read as arm elevation (tested, documented). The assumption that MediaPipe world Y is gravity-aligned (upright camera) is NOT yet validated on real data; tilted cameras bias all angles.
- **Revisit Conditions**: After live validation against a reference inclinometer; camera tilt estimation; clinical review of whether this mode is acceptable for any use beyond engineering tests.

---

## ADR-011: Movement-Aware Layer with Explicit Selection; Rotation Only Under a Protocol
- **Status**: Accepted
- **Context**: The pipeline tracked one metric (total humerothoracic elevation) for every exercise, so flexion, abduction and scaption were indistinguishable and no rotation or horizontal movement existed.
- **Decision**: Add `estimate_motion` with a per-movement definition (reference, plane/projection, sign, range, validity region, preferred view), a `MotionEstimate` contract with confidence/status/reasons, and a separate advisory `suggest_movement`. The exercise is chosen explicitly; the calibrated channel and repetition FSM follow the selected movement. Axial rotation is allowed only under the elbow-90 / arm-at-side protocol with wrist visible, capped at LOW_CONFIDENCE (refines ADR-006, does not reverse it). Pose-gated frames are not tracking loss.
- **Rationale**: Avoids relabelling one number for different exercises; refuses to fabricate angles outside a movement's definition; classification of ambiguous poses (forward 90° = flexion = horizontal adduction) is unreliable, so selection is explicit.
- **Consequences**: Confidence factors/tolerances are heuristic and unvalidated. IMU-dependent measurements remain documented requirements only.
- **Revisit Conditions**: Validation against a reference system; availability of an upper-arm IMU or second camera.
