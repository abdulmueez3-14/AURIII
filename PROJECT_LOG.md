# PROJECT_LOG.md — Canonical Project Log & Current State Summary
## AURI: Computer Vision & Biomedical Engine

---

## 1. Current State Summary

- **Current Phase**: Phase 7 complete + pipeline integration (Ready for Phase 8 / Phase 9)
- **Current Iteration**: Iteration 7.2 (Pipeline Integration)
- **Features Implemented & Verified**:
  - Clean repository isolation established at `c:\Users\abdul\Topics\medic 2026\AURI`.
  - Python 3.12 LTS virtual environment created and validated via `uv`.
  - Core scientific dependencies resolved, installed, and validated (`mediapipe 1.2.0`, `opencv-python 5.0.0.93`, `numpy 2.5.4`, `scipy 1.18.1`, `pytest 9.2.0`, `ruff 0.17.3`, `pydantic 2.14.1`, `matplotlib 3.11.2`).
  - MediaPipe Tasks Vision `PoseLandmarker` API verified and integrated.
  - Mandatory persistent agent instructions established (`AGENTS.md`).
  - Product requirements documented with traceable IDs (`PRD.md`).
  - Technology stack and dependency rationale documented (`TECH_STACK.md`).
  - Phased implementation roadmap established (`PHASES.md`).
  - Project configuration files (`pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md`) created.
  - Complete architectural, mathematical, and biomechanical documentation created in `docs/`:
    - `docs/ARCHITECTURE.md`
    - `docs/MATHEMATICAL_MODEL.md`
    - `docs/BIOMECHANICAL_CONVENTIONS.md`
    - `docs/VALIDATION_PLAN.md`
    - `docs/KNOWN_LIMITATIONS.md`
    - `docs/DECISIONS.md`
  - Synthetic ground-truth body fixtures implemented (`tests/synthetic/fixtures.py`).
  - Geometric exception hierarchy implemented (`src/auri_cv/geometry/exceptions.py`):
    - `GeometryError`, `DegenerateGeometryError`, `DegenerateVectorError`, `NonFiniteVectorError`, `InvalidRotationMatrixError`, `SingularProjectionError`.
  - Vector operations implemented with numerical guards (`src/auri_cv/geometry/vectors.py`).
  - Rotation matrix and transformation engine implemented (`src/auri_cv/geometry/rotations.py`).
  - Angular kinematics implemented (`src/auri_cv/geometry/angles.py`).
  - Pose observation data models implemented (`src/auri_cv/pose/data_models.py`).
  - Landmark quality gate implemented (`src/auri_cv/quality/landmark_quality.py`).
  - Production wrapper around MediaPipe Tasks `PoseLandmarker` implemented (`src/auri_cv/pose/landmarker.py`).
  - Torso Coordinate Frame estimator implemented (`src/auri_cv/frames/torso_frame.py`).
  - Anatomical segment estimation implemented (`src/auri_cv/anatomy/segments.py`):
    - Models upper-arm and forearm direction vectors strictly on the unit sphere ($\mathbb{S}^2$), preventing invalid axial rotation assumptions.
    - Computes upper arm direction $\hat{\mathbf{v}}_{\text{arm}}$ and forearm direction $\hat{\mathbf{v}}_{\text{forearm}}$ in both World and Torso frames.
    - Evaluates metric segment lengths ($L_{\text{arm}}, L_{\text{forearm}}$) with minimum threshold guards ($0.10$ m).
    - Computes anatomical elbow flexion angle ($0^\circ$ at full extension).
    - Robust handling when the wrist is occluded or missing.
  - Test suites implemented:
    - `tests/unit/test_segments.py` (straight hanging arm, 90° bent elbow, missing wrist, left arm laterality, degenerate segment rejection)
    - `tests/synthetic/test_segments.py` (verified segment directions on analytical ground truth postures: neutral, 90° abduction, 180° overhead, 90° forward flexion)
    - Full existing test suite
  - Package installed in editable mode (`auri-cv==0.1.0`).
- **Features Implemented but Not Verified**:
  - None.
- **Features Not Implemented**:
  - Humeral axial rotation feasibility assessment (Phase 8).
  - Full regression and reference validation suite (Phase 9).
- **Latest Test Results**:
  - `pytest`: 225 passed (all synthetic and unit tests).
  - `ruff check .`: All checks passed with zero errors or warnings.
  - `ruff format --check .`: 43 files checked, zero formatting violations.
- **Current Blockers**:
  - None.
- **Important Decisions**:
  - `ADR-001` through `ADR-009` maintained.
- **Known Limitations**:
  - Monocular depth ($Z$) from MediaPipe is an inferred heuristic rather than a physical distance measurement.
  - Collinear shoulder-to-elbow vector alone cannot observe humeral axial rotation.
- **Exact Next Task**:
  - Phase 8: humeral axial rotation feasibility study (`docs/AXIAL_ROTATION_FEASIBILITY.md`); then Phase 9: video runner script, real-video smoke test, noise-sensitivity benchmark, mypy/pyright.

---

## 2. Iteration History

### Iteration 0.1 — 2026-10-28
- **Objective**: Phase 0 Environment initialization, dependency resolution, API verification, and repository scaffolding.
- **Requirements Addressed**: `SYS-REQ-001`, `PRIV-REQ-001`.
- **Files Created / Modified**:
  - `.gitignore` (Created)
  - `pyproject.toml` (Created)
  - `requirements.txt` (Created)
  - `AGENTS.md` (Created)
  - `PRD.md` (Created)
  - `TECH_STACK.md` (Created)
  - `PHASES.md` (Created)
  - `PROJECT_LOG.md` (Created)
  - `README.md` (Created)
  - `docs/ARCHITECTURE.md` (Created)
  - `docs/MATHEMATICAL_MODEL.md` (Created)
  - `docs/BIOMECHANICAL_CONVENTIONS.md` (Created)
  - `docs/VALIDATION_PLAN.md` (Created)
  - `docs/KNOWN_LIMITATIONS.md` (Created)
  - `docs/DECISIONS.md` (Created)
  - `src/auri_cv/__init__.py` (Created)
  - `tests/__init__.py` (Created)
  - `tests/synthetic/__init__.py` (Created)
  - `tests/synthetic/fixtures.py` (Created)
  - `tests/synthetic/test_synthetic_definitions.py` (Created)
- **Changes Made**:
  - Verified that workspace `c:\Users\abdul\Topics\medic 2026\AURI` is clean and isolated from parent legacy files (`..\build` and `..\*`).
  - Created `.venv` utilizing CPython 3.12.10 via `uv`.
  - Installed all core scientific, testing, and linting dependencies via `uv pip`.
  - Tested MediaPipe 1.2.0 in Python to determine active API: verified that `mp.tasks.python.vision.PoseLandmarker` is the active API, while legacy `mp.solutions.pose` is removed.
  - Formulated persistent agent rules in `AGENTS.md` and product requirements in `PRD.md`.
  - Defined synthetic body fixtures with known analytical angles.
- **Decisions & Rationale**:
  - Selected Python 3.12 LTS over 3.14 to guarantee pre-compiled C-extension wheels across all scientific packages.
  - Pinned exact dependency versions in `requirements.txt` to guarantee deterministic reproducibility.
- **Commands & Tests Executed**:
  - `uv venv .venv --python 3.12`: Succeeded.
  - `uv pip install mediapipe opencv-python numpy scipy pytest ruff matplotlib pydantic`: Succeeded.
  - `pytest tests/synthetic/test_synthetic_definitions.py`: 5 passed in 0.18s.
- **Actual Results & Failures**:
  - No failures. All imports and synthetic tests passed.
- **Documentation Updated**:
  - All core documents created and cross-referenced.
- **Remaining Limitations**:
  - Phase 1 math primitives pending implementation.
- **Next Recommended Task**:
  - Proceed with Phase 1: Implement mathematical foundation.

---

### Iteration 1.1 — 2026-10-28
- **Objective**: Phase 1 Mathematical foundations, safe vector algebra, rotation matrix SO(3) validation, angular kinematics, and synthetic known-angle benchmarks.
- **Requirements Addressed**: `BIO-REQ-001`, `BIO-REQ-002`, `BIO-REQ-003`.
- **Files Created / Modified**:
  - `src/auri_cv/geometry/exceptions.py` (Created)
  - `src/auri_cv/geometry/vectors.py` (Created)
  - `src/auri_cv/geometry/rotations.py` (Created)
  - `src/auri_cv/geometry/angles.py` (Created)
  - `src/auri_cv/geometry/__init__.py` (Created)
  - `tests/unit/test_vectors.py` (Created)
  - `tests/unit/test_rotations.py` (Created)
  - `tests/unit/test_angles.py` (Created)
  - `tests/synthetic/test_angles.py` (Created)
  - `pyproject.toml` (Modified: configured pythonpath and lint rules)
  - `PHASES.md` (Modified: marked Phase 1 COMPLETE)
  - `PROJECT_LOG.md` (Modified: updated current state and history)
- **Changes Made**:
  - Built typed geometric exception hierarchy (`GeometryError`, `DegenerateVectorError`, `NonFiniteVectorError`, `InvalidRotationMatrixError`, `SingularProjectionError`).
  - Implemented safe vector algebra in `vectors.py`: magnitude, normalization with degenerate safeguards ($\epsilon = 10^{-7}$), dot product, cross product, safe cosine with strict $[-1.0, 1.0]$ clamping, and plane projection.
  - Implemented rotation validation in `rotations.py`: SO(3) check asserting $\|R^T R - I\|_F < 10^{-5}$ and $\det(R) = +1.0 \pm 10^{-5}$, frame transformation functions for points and vectors with strict separation (vectors never translated), axis-angle generation, and Hamilton scalar-first quaternion conversions.
  - Implemented angular kinematics in `angles.py`: unsigned vector angle with domain clamping $[0, \pi]$, signed angle around arbitrary 3D normal axis with right-hand rule projection and $\operatorname{atan2}$, degree/radian conversions, and angle wrapping.
  - Created 77 new unit and synthetic test cases covering orthogonal vectors, anti-parallel vectors, degenerate vectors, NaN/Inf vectors, SO(3) matrix validity, point/vector transformations, quaternion round-trips, and known analytical angles (0°, 30°, 45°, 60°, 90°, 120°, 135°, 150°, 180°) across planar and arbitrary 3D inclined orientations.
  - Installed `auri-cv` in editable mode (`uv pip install -e .`).
- **Decisions & Rationale**:
  - Configured Ruff to allow uppercase $R$ for rotation matrices and $I$ for identity matrices in mathematical modules in alignment with standard biomechanical linear algebra conventions.
  - Enforced strict numerical tolerance ($10^{-5}$) for SO(3) matrix orthonormality and determinant checks.
- **Commands & Tests Executed**:
  - `uv pip install -e .`: Successfully installed `auri-cv==0.1.0`.
  - `ruff check .`: Passed with zero errors.
  - `ruff format --check .`: 26 files checked, zero formatting violations.
  - `pytest`: 82 passed in 6.23s.
- **Actual Results & Failures**:
  - Initial import failure due to `src` layout was resolved cleanly by configuring `pythonpath = ["src"]` in `pyproject.toml` and installing `auri-cv` in editable mode.
- **Documentation Updated**:
  - `PHASES.md`, `PROJECT_LOG.md`.
- **Remaining Limitations**:
  - Pose estimation ingestion and torso frame estimation modules are pending in Phases 2 and 3.
- **Next Recommended Task**:
  - Phase 2: Pose Ingestion, Normalization & Quality Gate.

---

### Iteration 2.1 — 2026-10-28
- **Objective**: Phase 2 Pose ingestion, landmark normalization, quality gates, and MediaPipe Tasks PoseLandmarker integration.
- **Requirements Addressed**: `CV-REQ-001`, `CV-REQ-002`, `CV-REQ-003`, `CV-REQ-004`, `CV-REQ-005`, `QUAL-REQ-001`, `PRIV-REQ-001`, `PRIV-REQ-002`.
- **Files Created / Modified**:
  - `src/auri_cv/pose/data_models.py` (Created)
  - `src/auri_cv/pose/landmarker.py` (Created)
  - `src/auri_cv/pose/__init__.py` (Created)
  - `src/auri_cv/quality/landmark_quality.py` (Created)
  - `src/auri_cv/quality/__init__.py` (Created)
  - `tests/unit/test_pose_models.py` (Created)
  - `tests/unit/test_landmark_quality.py` (Created)
  - `tests/unit/test_landmarker.py` (Created)
  - `tests/synthetic/test_quality_gates.py` (Created)
  - `PHASES.md` (Modified: marked Phase 2 COMPLETE)
  - `PROJECT_LOG.md` (Modified: updated current state and history)
- **Changes Made**:
  - Downloaded and verified MediaPipe Tasks PoseLandmarker model bundle (`models/pose_landmarker_lite.task`).
  - Created typed, frozen Pydantic data models for `Landmark2D`, `Landmark3D`, `LandmarkIndex`, and `PoseObservation`, preserving both normalized 2D image coordinates and metric 3D world landmarks without loss.
  - Created `LandmarkQualityGate` implementing discrete multi-state quality assessment (`VALID`, `VALID_WITH_WARNING`, `UNAVAILABLE`, `RECALIBRATION_REQUIRED`), enforcing visibility threshold ($V_{\min} = 0.5$) and presence threshold ($P_{\min} = 0.5$) on key torso and target arm landmarks, plus coordinate finiteness checks.
  - Built `AuriPoseLandmarker` production wrapper supporting `IMAGE` and `VIDEO` running modes, input video mirroring, automatic model asset discovery/download, and clean context-manager C++ resource lifecycle management.
  - Implemented 20 new unit and synthetic test cases verifying data model validation, pixel coordinate mapping, quality degradation on missing/occluded landmarks, target arm isolation, and MediaPipe inference on blank frames.
- **Decisions & Rationale**:
  - Used `enum.StrEnum` for `QualityStatus` to guarantee serialization compatibility with both Pydantic schemas and string displays.
  - Isolated quality assessment of the target exercise arm from the contralateral arm so that an occluded non-target arm does not falsely invalidate the active exercise measurement.
- **Commands & Tests Executed**:
  - `pytest -q`: 102 passed in 3.83s.
  - `ruff check .`: Passed with zero errors.
  - `ruff format --check .`: 35 files checked, zero formatting violations.
- **Actual Results & Failures**:
  - None. MediaPipe Tasks inference and landmark quality gates executed deterministically.
- **Documentation Updated**:
  - `PHASES.md`, `PROJECT_LOG.md`.
- **Remaining Limitations**:
  - Torso frame estimation and segment vector construction are pending in Phases 3 and 4.
- **Next Recommended Task**:
  - Phase 3: Torso Coordinate Frame Estimation (`src/auri_cv/frames/torso_frame.py`).

---

### Iteration 3.1 — 2026-10-28
- **Objective**: Phase 3 Torso Coordinate Frame Estimation, Gram-Schmidt orthogonalization, proper SO(3) validation, degenerate torso rejection, and synthetic perturbation invariance.
- **Requirements Addressed**: `BIO-REQ-004`, `BIO-REQ-005`, `BIO-REQ-006`.
- **Files Created / Modified**:
  - `src/auri_cv/geometry/exceptions.py` (Modified: added `DegenerateGeometryError`)
  - `src/auri_cv/geometry/__init__.py` (Modified: exported `DegenerateGeometryError`)
  - `src/auri_cv/frames/torso_frame.py` (Created)
  - `src/auri_cv/frames/__init__.py` (Created)
  - `tests/unit/test_torso_frame.py` (Created)
  - `tests/synthetic/test_torso_frame.py` (Created)
  - `PHASES.md` (Modified: marked Phase 3 COMPLETE)
  - `PROJECT_LOG.md` (Modified: updated current state and history)
- **Changes Made**:
  - Built `TorsoFrame` dataclass storing the anatomical basis unit vectors ($\hat{\mathbf{x}}_T$ Right, $\hat{\mathbf{y}}_T$ Anterior, $\hat{\mathbf{z}}_T$ Superior), origin $\mathbf{o}_T$ at shoulder midpoint, forward $R_{W \leftarrow T}$ and inverse $R_{T \leftarrow W}$ rotation matrices, shoulder width, and torso length.
  - Implemented `estimate_torso_frame`:
    - Computes midpoints $\mathbf{S}_c$ and $\mathbf{H}_c$.
    - Constructs superior axis $\hat{\mathbf{z}}_T$ headward along spine.
    - Applies Gram-Schmidt orthogonalization on lateral vector candidate $\mathbf{x}_0 = \mathbf{S}_R - \mathbf{S}_L$ to eliminate spine component, yielding pure lateral axis $\hat{\mathbf{x}}_T$.
    - Derives anterior axis $\hat{\mathbf{y}}_T = \hat{\mathbf{z}}_T \times \hat{\mathbf{x}}_T$ adhering strictly to right-handed Cartesian rules ($\hat{\mathbf{x}}_T \times \hat{\mathbf{y}}_T = \hat{\mathbf{z}}_T$).
    - Asserts SO(3) matrix validity ($\|R^T R - I\|_F < 10^{-5}$ and $\det(R) = +1.0 \pm 10^{-5}$).
  - Implemented explicit numerical guards rejecting degenerate geometries: shoulder width $< 0.15$ m, hip width $< 0.15$ m, spine length $< 0.15$ m, or collinear shoulder-spine configurations.
  - Added point and direction vector transformation methods ensuring direction vectors are never translated.
  - Implemented 30 new unit and synthetic test cases:
    - `tests/unit/test_torso_frame.py`: tested standard upright posture, point mappings, direction transformations, and degenerate failure cases.
    - `tests/synthetic/test_torso_frame.py`: tested 20 combinations of synthetic postures under translation, yaw, pitch, roll, and combined 6-DoF perturbations, proving mathematical invariance of torso-relative arm coordinates within $10^{-5}$ meters.
- **Decisions & Rationale**:
  - Retained shoulder center midpoint $\mathbf{S}_c$ as origin $\mathbf{o}_T$ so that torso-relative coordinates directly express glenohumeral girdle excursions.
  - Enforced strict minimum physical span thresholds ($15$ cm) to prevent numerical explosions when landmarks collapse or overlap.
- **Commands & Tests Executed**:
  - `pytest -q`: 132 passed in 4.06s.
  - `ruff check .`: Passed with zero errors.
  - `ruff format --check .`: 39 files checked, zero formatting violations.
- **Actual Results & Failures**:
  - None. All synthetic perturbation invariance tests matched within $10^{-5}$ tolerance.
- **Documentation Updated**:
  - `PHASES.md`, `PROJECT_LOG.md`.
- **Remaining Limitations**:
  - Upper-arm and forearm segment direction construction pending in Phase 4.
- **Next Recommended Task**:
  - Phase 4: Anatomical Segment Estimation (`src/auri_cv/anatomy/segments.py`).

---

### Iteration 4.1 — 2026-10-28
- **Objective**: Phase 4 Anatomical Segment Estimation, S^2 unit direction vectors, metric lengths, and anatomical elbow flexion.
- **Requirements Addressed**: `BIO-REQ-007`.
- **Files Created / Modified**:
  - `src/auri_cv/anatomy/segments.py` (Created)
  - `src/auri_cv/anatomy/__init__.py` (Created)
  - `tests/unit/test_segments.py` (Created)
  - `tests/synthetic/test_segments.py` (Created)
  - `PHASES.md` (Modified: marked Phase 4 COMPLETE)
  - `PROJECT_LOG.md` (Modified: updated current state and history)
- **Changes Made**:
  - Built `ArmSegmentState` dataclass strictly enforcing unit direction vectors ($\mathbb{S}^2$) rather than full $\text{SO}(3)$ orientations, ensuring that unobservable humeral axial rotation cannot be falsely inferred from a 1D segment.
  - Implemented `estimate_arm_segments` and `estimate_arm_segments_from_pose`:
    - Constructs normalized upper-arm vector $\hat{\mathbf{v}}_{\text{arm}}$ (shoulder to elbow) in World and Torso frames.
    - Constructs optional normalized forearm vector $\hat{\mathbf{v}}_{\text{forearm}}$ (elbow to wrist) in World and Torso frames.
    - Computes metric segment lengths with numerical safeguard against collapsed segments ($< 0.10$ m).
    - Computes anatomical elbow flexion angle ($0^\circ$ at straight extension, $90^\circ$ at right angle).
    - Gracefully handles occluded wrist while maintaining valid upper-arm kinematics.
  - Implemented 12 new unit and synthetic test cases:
    - `tests/unit/test_segments.py`: verified straight arm, 90° bent elbow, missing wrist fallback, left arm laterality, and degenerate segment rejection.
    - `tests/synthetic/test_segments.py`: verified analytical segment unit vectors across neutral ($[0, 0, -1]$), 90° abduction ($[1, 0, 0]$), 180° overhead ($[0, 0, 1]$), and 90° forward flexion ($[0, 1, 0]$).
- **Decisions & Rationale**:
  - Retained strict boundary guarantee: segment representations are $\mathbb{S}^2$ unit vectors, adhering to `ADR-006` and preventing arbitrary or misleading rotation matrix assignments.
  - Used `np.isclose` with absolute tolerance for floating-point metric length checks.
- **Commands & Tests Executed**:
  - `pytest -q`: 144 passed in 4.08s.
  - `ruff check .`: Passed with zero errors.
  - `ruff format --check .`: 43 files checked, zero formatting violations.
- **Actual Results & Failures**:
  - Resolved exact float equality assertion on metric segment length by applying `np.isclose`.
- **Documentation Updated**:
  - `PHASES.md`, `PROJECT_LOG.md`.
- **Remaining Limitations**:
  - Joint angle kinematics (elevation, sagittal flexion, coronal abduction) pending in Phase 5.
- **Next Recommended Task**:
  - Phase 5: Joint-Angle Kinematics Solver (`src/auri_cv/kinematics/elevation.py`, `projections.py`).

---

### Iteration 5.1 — 2026-10-29 (takeover from Antigravity)
- **Objective**: Complete Phase 5. Audit found `kinematics/elevation.py` and `projections.py` already written by the previous agent but with no package `__init__`, no tests, 2 ruff errors, and PHASES/PROJECT_LOG still saying NOT_STARTED.
- **Requirements Addressed**: `BIO-REQ-007`, `BIO-REQ-008`.
- **Files Created / Modified**: `kinematics/__init__.py`, `kinematics/solver.py` (new: `RawKinematics`, `solve_arm_kinematics`); `kinematics/projections.py` (degenerate-plane guard); `kinematics/elevation.py` (import order); `tests/synthetic/test_kinematics.py` (new); `docs/DECISIONS.md` (ADR-008); `docs/VALIDATION_PLAN.md`; `PHASES.md`.
- **Changes Made**: Found that flexion/abduction/plane-of-elevation returned rounding-noise values (0 vs 180 deg) or a false 0 when the arm has no component in the plane; now raise `DegenerateVectorError`, mapped to `None` in `RawKinematics` (ADR-008). Added tests for analytic angles (0–180 deg, planes 0/45/90), signs, laterality symmetry, degenerate/NaN input, fixture poses end-to-end.
- **Commands & Tests Executed**: `ruff format .`, `ruff check .` (clean); `pytest -q`: 172 passed.
- **Remaining Limitations**: 5 deg in-plane threshold is an engineering default. `docs/MATHEMATICAL_MODEL.md` not yet extended with the projection formulas (only elevation is documented there). No mypy/pyright run configured.
- **Next Recommended Task**: Phase 6 calibration and neutral-zone deadband.

---

### Iteration 6.1 — 2026-10-29
- **Objective**: Phase 6 baseline calibration, relative excursion, neutral-zone deadband with hysteresis, failure transparency.
- **Requirements Addressed**: `CAL-REQ-001` to `CAL-REQ-005`, `QUAL-REQ-002`; `CAL-REQ-006` partially.
- **Files Created**: `calibration/calibrator.py` (`BaselineCalibrator`: median of N=30 consecutive valid frames, IQR gate 3 deg, gaps restart collection), `calibration/neutral_zone.py` (`NeutralZone`: enter 3 / exit 5 deg hysteresis, display-only), `calibration/session.py` (`CalibratedChannel`: raw/relative/display kept side by side; statuses CALIBRATING, CALIBRATION_REJECTED, OK, UNAVAILABLE, RECALIBRATION_REQUIRED latched), `calibration/__init__.py`, `tests/synthetic/test_calibration.py` (18 tests).
- **Decisions**: Channel is generic over one scalar angle (elevation, flexion, ...), so each tracked angle gets its own calibrated channel. Hysteresis: leave neutral at |d| >= exit, re-enter at |d| <= enter (matches MATHEMATICAL_MODEL 7.3). Non-usable quality or None raw yields `display_deg=None`, never 0.0. Tracking gap > 15 frames latches RECALIBRATION_REQUIRED (tunable default).
- **Tests**: `ruff format`, `ruff check` clean; `pytest -q`: 190 passed.
- **Remaining Limitations**: Subject relocation / torso-frame shift detection not implemented (MediaPipe world landmarks are hip-centred, so translation is not observable; an orientation/scale-based monitor is needed). Deadband thresholds are PRD defaults, not clinically validated (PRD Open Question 1). Not yet wired to `RawKinematics`/pose end to end.
- **Next Recommended Task**: Phase 7.

---

### Iteration 7.1 — 2026-10-29
- **Objective**: Phase 7 temporal filters, repetition FSM, compensation metrics.
- **Requirements Addressed**: `REP-REQ-001` to `REP-REQ-004`, `COMP-REQ-001` to `COMP-REQ-004`.
- **Files Created**: `filtering/filters.py` (`MovingAverageFilter`, `ExponentialFilter`; gaps clear state), `movements/repetition_fsm.py` (`RepetitionFSM`, `Repetition`), `movements/compensation.py` (`compute_compensation`, `CompensationMetrics`, `CompensationWarning`), package `__init__`s, `tests/synthetic/test_repetition_fsm.py`, `tests/synthetic/test_compensation.py` (27 tests).
- **Bug found by tests**: after a gap-invalidated rep the FSM resumed mid-descent and counted a bogus rep with time-to-peak 0. Added arming rule: a rep can only start after the signal was seen <= 10 deg (also prevents counting when a session starts with the arm already raised). ADR-009.
- **Decisions**: Lean and rotation are measured against the calibration-baseline TorsoFrame, not a gravity vector (ADR-009). Gap > 1.0 s in RAISING/PEAK_HOLD/LOWERING invalidates the rep (spec lists RAISING/LOWERING; PEAK_HOLD added for consistency with REP-REQ-003).
- **Tests**: `ruff format`, `ruff check` clean; `pytest -q`: 217 passed.
- **Remaining Limitations**: FSM thresholds (15/45/10/5 deg) are model-doc defaults and assume a unidirectional exercise whose excursion is positive; a second rise during LOWERING returns to PEAK_HOLD. Trunk rotation is relative to baseline, not the camera axis. No warning is emitted for trunk rotation (no spec threshold). Components are not yet composed into one per-frame pipeline.
- **Next Recommended Task**: Phase 8 feasibility document; then pipeline integration.

---

### Iteration 7.2 — 2026-10-29 (Pipeline integration)
- **Objective**: Compose all engine stages into one per-frame, I/O-free pipeline and test it end to end on synthetic pose sequences.
- **Files Created**: `src/auri_cv/pipeline.py` (`ShoulderAnalysisPipeline`, `FrameResult`), `tests/synthetic/pose_builder.py` (analytic `PoseObservation` sequences), `tests/synthetic/test_pipeline.py` (8 tests).
- **Behaviour**: quality gate -> torso frame -> segments -> solver -> EMA -> calibrated channel -> repetition FSM; compensation vs. the torso frame captured on the frame that completed calibration. `FrameResult` carries raw solver output, filtered elevation, channel reading (relative + display) and quality side by side. Tracked channel is total humerothoracic elevation; flexion/abduction/plane are in `raw`. Degenerate torso/segment geometry is treated as lost tracking.
- **Tests**: `ruff format`, `ruff check` clean; `pytest -q`: 225 passed. Covered: calibration then exact 0.0 display; two reps with peaks 90/120 deg; flexion plane; seeded 3 mm landmark noise; 2 s dropout (display None, latched RECALIBRATION_REQUIRED, no rep counted, `recalibrate()` restarts); lean 18 deg and hiking 5 cm warnings; low visibility. A mutation check (removing both the quality gating and the None-propagation) made the dropout and low-visibility tests fail, confirming they detect fabricated zeros.
- **Remaining Limitations**: Never run on real video or a real person: the real MediaPipe output (Y-down world frame, noise, depth error) is untested end to end. No video runner script. The calibration frame doubles as the compensation baseline, so a subject leaning during calibration is not flagged. Calibration phase has no user prompt/UI (out of scope). No type checker (mypy/pyright) configured.
- **Next Recommended Task**: Phase 8 feasibility document, then Phase 9.

---

### Iteration 7.3 — 2026-10-29 (First live webcam session, subject: project owner)
- **Objective**: First run of the full engine on real camera data.
- **Changes**: `scripts/run_camera.py` (live runner: mirrored preview + overlay, unmirrored frames to the model so L/R labels stay anatomical, numeric CSV only to `runs/`, no video/images saved; `.gitignore` updated). **Bug fixed in `pose/landmarker.py`**: `getattr(lm, "visibility", 1.0) or 1.0` converted a real 0.0 visibility/presence into 1.0 (a hidden joint would pass the quality gate); now `_score()`, regression test added. `pytest`: 226 passed.
- **Session result (single 75 s run, 1805 frames, 23.8 fps, median latency 26 ms)**: pose detected 99%, but hips were below the visibility threshold for most of the run (subject too close/seated at first), so only 231 frames gave valid calibrated readings and no reading was shown otherwise (gate behaved as designed; no fake 0 deg). Baseline was set at 25.3 deg raw elevation (IQR 2.2) with the arm presumably hanging; 1 repetition counted (peak 52 deg relative, ~77 deg raw), 1 invalidated by tracking loss; raw elevation up to 104 deg.
- **Not conclusions**: LEAN_DETECTED / HIKING_DETECTED fired only on frames with low-visibility elbow/wrist/hips and are not trustworthy; the 25 deg resting elevation cannot be attributed to the subject's posture vs. landmark/torso-frame bias (no ground truth); during a raise whose abduction ~ elevation, flexion read ~47 deg, which could be a real oblique plane or monocular depth error. No accuracy claim is made.
- **Finding / decision needed**: the torso frame requires both hips, which are often out of view for seated or close-range rehab use. Options: a shoulder-only fallback frame (less robust, no trunk lean from hips), or require standing framing.
- **Next**: repeat the session with proper framing (standing ~2 m, hips visible) and a known reference (e.g. a protractor/phone-inclinometer check at 90 deg).

---

### Iteration 7.4 — 2026-10-29 (Fixes after owner feedback on live session)
- **Feedback**: left arm appeared undetected; on-screen text was hard to read/overlapping; overall the session did not reach its goal.
- **Findings**: (1) the model DID detect the left arm (shoulder vis 0.996, elbow 0.73, wrist 0.55) -- the runner only drew the tracked arm, so it looked undetected (runner defect). (2) text was drawn directly on the camera image. (3) The real blocker: both hips ~0.01 visibility, so no torso frame and no angles.
- **Changes**: runner now draws both arms + torso, colours each landmark/bone by visibility (red < 0.5), text on solid panels, `--arm both` (one pipeline per arm, per-arm CSV rows and summary), `--torso hips|shoulders`. New `TorsoMode.SHOULDERS_ONLY_ASSUMED_VERTICAL` (`estimate_torso_frame_shoulders_only`, gate `require_hips=False`, `TorsoFrame.hips_observed`) -- ADR-010. 3 new tests.
- **Verification**: `ruff` clean; `pytest`: 229 passed. Overlay layout checked by rendering a synthetic pose to a PNG and inspecting it (no camera frames saved). Live validation NOT done: nobody was in front of the camera for the follow-up check, and the assumed-vertical Y-down convention remains unverified.
- **Next**: re-run live with `--arm both --torso shoulders` (and again with `--torso hips` standing 2 m back), compare against a protractor/inclinometer reading at 90 deg.

---

### Iteration 7.5 — 2026-10-29 (Second live session: `--arm both --torso shoulders`, project owner)
- **Result**: 1489 frames per arm, ~19.7 fps, median latency ~24 ms. Pose detected 100%; valid calibrated readings 1459/1489 (right) and 1458/1489 (left) -- tracking works for both arms in shoulders-only mode (previous session: 231/1805). Baselines: right 9.6 deg (IQR 0.86), left 12.4 deg (IQR 0.59). Raw elevation ranges: right 8..144.5 deg, left 5..100.6 deg. Hanging left arm read a steady ~10 deg elevation / ~-6 deg flexion for 25 s (stable, but no ground truth to separate true posture from bias).
- **Not established**: accuracy (no protractor/inclinometer reference supplied); the assumed-vertical (ADR-010) validity; what the subject saw on screen (overlay legibility) -- awaiting feedback.
- **Repetition counting is unreliable for free-form movement**: counted 4 (right) / 2 (left) "reps" with durations up to 32 s and time-to-peak 24.8 s, because the arm never returned below the 10 deg return threshold between slow/continuous movements. The FSM assumes discrete raise-and-return cycles; a structured protocol (or a minimum-duration/hold rule) is needed before rep metrics are meaningful.
- **No compensation warnings** (disabled in this mode by design).
- **Next**: structured protocol with a 90 deg inclinometer check; revisit FSM for slow/continuous movement; Phase 8/9.

---

### Iteration 7.6 — 2026-10-29 (Third live session: hold/validation attempt, shoulders-only mode)
- **Observed (right arm, log `runs/session_20261029_143244.csv`)**: rest 15-20 deg elevation (earlier session: ~10 deg); a 12 s hold at elev ~78 deg, abduction ~73, flexion ~74, plane ~45 deg (internally consistent: atan2 identity holds), within-hold spread ~2-5 deg; later plateaus at elev ~111 deg (plane ~0) and ~72 deg (plane ~45). Left arm stayed at ~9-12 deg elevation throughout (not raised); left channel latched RECALIBRATION_REQUIRED for 285 frames after a short dropout (~50-52 s).
- **Not established**: error vs. true angle. The subject had no inclinometer/protractor and the poses performed did not clearly match the requested order (no plateau at plane ~0 or ~90 at 90 deg elevation), so no reference-based error could be computed. Resting reading varied ~6-10 deg between sessions for the same arm.
- **Next**: obtain the actual pose performed at each hold (and any reference reading) before claiming any accuracy.

---

### Iteration 8.1 — 2026-10-29 (Movement-aware shoulder analysis)
- **Request**: extend beyond a single metric to flexion, extension, abduction, adduction, internal/external rotation, horizontal abduction/adduction, scaption (+ optional circumduction), with movement-specific angles, confidence, tests, UI labels.
- **Audit / root cause**: the engine tracked total humerothoracic elevation for every exercise (not a rotation); flexion/abduction existed only as raw projections; no movement layer, no validity regions, rotation rejected by ADR-006. This repo has no IMU, dashboard or simulator code (out of scope per AGENTS.md; IMU contract is documentation only).
- **Changes**: `movements/shoulder_movements.py` (`ShoulderMovement`, `MovementSpec`/`SPECS`, `estimate_motion`, `MotionEstimate`, `suggest_movement`, `CameraAxes`, `estimate_subject_yaw_deg`, view classification, heuristic confidence); `movements/circumduction.py`; pipeline `movement=` option, `FrameResult.motion/.suggestion`; `CalibratedChannel.update(pose_gated=)` (pose-gated frames do not latch recalibration); runner `--movement`, overlay labels, CSV columns, summary; test builder gained left arm / forearm direction / body yaw; docs `MOVEMENT_ANALYSIS.md`, ADR-011, PRD `MOV-REQ-001..006`, KNOWN_LIMITATIONS, PHASES (Phase 8 IN_PROGRESS).
- **Verification**: `ruff format --check` and `ruff check` clean; `pytest -q`: 344 passed (was 229; +89 per-movement unit/synthetic, +26 end-to-end). Mutation check (left-arm rotation sign, extension sign, off-plane guard each broken on purpose) failed 3, 8 and 2 tests respectively. Runner overlay/CSV verified on synthetic frames only. No mypy/pyright is installed (not in the approved stack), so no static type check was run. Live camera test of the new movement modes NOT done.
- **Not validated**: all angles are geometry on synthetic data; confidence factors/tolerances are uncalibrated; rotation and horizontal movements are depth-sensitive; shoulders-only vertical assumption unverified on real data.
- **Next**: live runs with `--movement` against physical references; decide recovery policy for the recalibration latch (open question to owner); consider second-camera/IMU requirements for rotation and horizontal movements.

---

### Iteration 8.2 — 2026-10-29 (First live run of movement-specific mode: `--movement abduction --arm both --torso shoulders`)
- **Result** (`runs/session_20261029_151159.csv`, 1192 frames/arm, ~19.7 fps, latency ~26 ms): camera view estimated FRONTAL (yaw 1-5 deg) for ~95% of frames; rest abduction ~11 deg (right) / ~8 deg (left); brief passes at ~95-105 deg and ~144-152 deg (confidence 0.6); a held ~48 deg segment (conf 0.9).
- **NOT_MEASURABLE share**: 413/1192 (right) and 448/1192 (left) frames. For ~20 s both arms were at 65-90 deg elevation with 27-35 deg anterior component ("out of the coronal plane"), so abduction was withheld by design (25 deg tolerance). Unknown whether the arms were truly forward of the coronal plane or whether monocular depth error produced the anterior component; no reference. Both arms showed the same offset.
- **Not established**: accuracy; whether the 25 deg tolerance is too strict for real "arms out to the side" movement (patients often elevate in the scapular plane). No change made.
- **Observed**: ~60 frames of elbow/wrist dropout at the end (right-arm elbow/wrist visibility), rep counts again not meaningful for free-form movement.

---

### Iteration 8.3 — 2026-10-29 (Guided all-movements session, first live run)
- **Owner feedback**: in the previous abduction run the arms were held FORWARD, so withholding abduction ("arm out of the coronal plane") was correct. Owner asked for all features to be calculated.
- **Added**: `ShoulderAnalysisPipeline(compute_all_movements=True)` -> `FrameResult.all_motions` (all nine estimates per frame, each against its own definition); runner `--guided` (ten prompted steps, per-movement CSV columns `ang_*`/`st_*`, prompted-movement coverage table) and `--all-movements`. Tests: 347 passed.
- **Live run** (`runs/session_20261029_151733.csv`, 2105 frames/arm, ~23 fps, 91 s of steps): prompted movements were mostly NOT performed as defined: arm already raised during CALIBRATE (71 deg); ABDUCTION step arm barely moved (max 27 deg); steps 6/7/9/10 show the arm held out to the side (~65-86 deg elevation, plane ~0, i.e. abduction) instead of the prompted scaption / horizontal / rotation poses; rotation steps had elbow flexion 15-35 deg (protocol requires 70-110), so rotation was correctly NOT_MEASURABLE (0% of frames). Peaks of prompted metrics were low (flexion 2, extension 9, abduction 26, scaption 35-40, horizontal adduction 10-34). The prompts likely moved too fast to follow. Coverage "measurable" is not evidence of movement for planar metrics (a hanging arm is measurable at 0 deg).
- **Fixes after the run**: reason ordering (the generic "confidence is a heuristic" note no longer masks the real cause; UNRELIABLE now leads with "below reporting floor"); guided protocol now has a 5 s "get ready" lead-in per step (never scored), beep on phase change, larger prompt text, 6-12 s action windows (160 s total).
- **Not established**: accuracy of any movement angle; no reference. Rotation and horizontal movements remain unmeasured in practice.
