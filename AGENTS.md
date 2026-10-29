# AGENTS.md — Persistent Coding-Agent Operating Rules
## AURI Computer Vision & Biomedical Engine

This document defines the strict, non-negotiable rules for all AI coding agents working on this codebase. Reread this document at the start of every session.

---

## 1. Project Mission & Scope
- **Mission**: Build a standalone, modular, testable, explainable Python system for camera-based upper-limb movement analysis, focused initially on adhesive capsulitis (frozen shoulder) rehabilitation.
- **Current Scope**: Strictly computer vision, 2D/3D pose interpretation, coordinate geometry, joint kinematics, baseline calibration, repetition counting, compensation detection, and measurement quality gates.
- **Out of Scope**: Mobile applications, Flutter, Dart, Android Studio, Kotlin, BLE device connections, cloud services, and clinician dashboards. Document future interfaces only; do not implement them.
- **Legacy Boundary**: The legacy Shoulder ROM Tracker in adjacent directories is strictly read-only and outside this repository's write scope. Never copy legacy application code into this repository.

---

## 2. Clinical Boundaries & Safety
- **Non-Diagnostic**: AURI is a rehabilitation-support and kinematic measurement tool. It does **not** independently diagnose frozen shoulder, determine clinical stage, prescribe treatment, or override clinical directives.
- **No False Claims**: Never claim "clinical-grade accuracy" or "clinical validation" without peer-reviewed empirical evidence. Distinguish clearly between geometric angle estimates and anatomical joint angles.
- **Safety**: Flag excessive compensation, abnormal tracking loss, or extreme excursions. Do not generate clinical guidance without physician review.

---

## 3. Architecture & Modular Boundaries
The processing pipeline must strictly adhere to the following sequence:
```text
Video Acquisition 
  → Pose Estimation 
  → Landmark Normalization & Quality Gate 
  → Torso-Frame Estimation 
  → Segment Estimation 
  → Joint-Angle Solver 
  → Calibration & Neutral Zone 
  → Movement Analysis & Repetition Detection 
  → Measurement Quality Gate 
  → Results & Reporting
```
- Every stage must have a single defined responsibility, a typed input/output contract, isolated tests, and documented limitations.
- Keep mathematical calculations completely decoupled from video acquisition, GUI/OpenCV overlays, and file I/O.
- Raw observations must flow alongside interpreted results; never overwrite raw sensor/landmark data.

---

## 4. Mathematical Conventions & Safeguards
- **Coordinate Systems**: Every point, vector, and matrix must specify its frame of reference ($Image_{2D}$, $World_{3D}$, $Torso_T$, $Segment_A$).
- **Torso Frame ($T$)**:
  - $+X_T$: Subject's anatomical right.
  - $+Y_T$: Anterior (forward from torso).
  - $+Z_T$: Superior (headward along spine axis).
  - Handedness: Strictly right-handed ($\hat{x}_T \times \hat{y}_T = \hat{z}_T$).
- **Angles & Units**:
  - Internal calculations: Radians and SI units (meters, seconds, rad/s).
  - External boundary interfaces: Degrees ($^\circ$) and Hertz.
- **Numerical Guards**:
  - Cosine clamping: Always clamp dot products to $[-1.0, 1.0]$ before calling $\arccos$.
  - Degenerate checks: Test vectors for zero/near-zero length ($\|\mathbf{v}\| < 10^{-6}$) and non-finite values (NaN, Inf) before normalizing.
  - Rotation matrix validation: Assert $R^T R = I$ and $\det(R) = +1.0$ within numerical tolerance ($10^{-5}$).
  - Never apply point translation to direction vectors.
  - Never calculate humeral axial rotation from a single collinear shoulder-to-elbow vector.

---

## 5. Calibration, Neutral Zone & Display Deadband
- **Calibration**: Collect $N$ stable frames in clinician-approved starting posture. Set median as reference $\theta_0$.
- **Relative Excursion**: $\Delta\theta(t) = \theta(t) - \theta_0$.
- **Display Deadband**: If $|\Delta\theta(t)| \le \delta_{exit}$, display $0.0^\circ$. Apply hysteresis ($\delta_{enter} < \delta_{exit}$) to eliminate display flicker.
- **Data Integrity**: The deadband applies to the **display layer only**. Never zero out raw or filtered kinematic data.
- **Failure Transparency**: Never display $0^\circ$ when tracking fails. If landmarks are lost or frame is degenerate, emit explicit status (`UNAVAILABLE`, `RECALIBRATION_REQUIRED`).

---

## 6. Coding & Tooling Standards
- **Language**: Python 3.12 LTS.
- **Typing**: 100% type-annotated (`mypy` / `pyright` compliant). Use Pydantic or typed dataclasses for data contracts.
- **Linting & Formatting**: Ruff with strict rules. Line length 100 characters.
- **Testing**: `pytest`. All math and geometry functions must pass synthetic tests with known analytical solutions before camera data is processed.
- **Zero Mocking of Math**: Never mock numerical calculations. Compute exact geometric results in tests.

---

## 7. Workflow & Agent Protocol
At the start of every session/task:
1. Read `AGENTS.md`.
2. Read current status in `PROJECT_LOG.md`.
3. Check roadmap in `PHASES.md`.
4. Inspect relevant code and test files.
5. Identify the active phase and acceptance criteria.

During implementation:
- Make atomic, reviewable commits/changes.
- Do not add unapproved dependencies.
- Verify version compatibility with installed packages.
- Test failure cases explicitly.

After every iteration:
- Append an entry to `PROJECT_LOG.md` (date, iteration ID, changes, tests run, decisions, next steps).
- Update `PHASES.md` status if phase criteria are met.
- Update `TECH_STACK.md` if dependencies change.
- Update `docs/DECISIONS.md` when architectural decisions are made.

---

## 8. State Recovery
If context is interrupted or lost:
1. Run `git status` or inspect directory contents to identify active branch/files.
2. Check `PROJECT_LOG.md` "Current State Summary" to identify the active phase, latest test results, and next task.
3. Run `pytest` to verify the health of the test suite.
4. Resume exactly at the active phase documented in `PHASES.md`.

---

## 9. Definition of Done
A phase or feature is DONE only when:
1. Requirements are mapped to specific requirement IDs.
2. Mathematical formulations and assumptions are documented in `docs/MATHEMATICAL_MODEL.md`.
3. Data contracts are defined with strict types.
4. Unit and synthetic tests pass with 100% determinism.
5. Failure modes, edge cases, and degenerate states are explicitly handled.
6. Known limitations are recorded in `docs/KNOWN_LIMITATIONS.md`.
7. `PROJECT_LOG.md` and `PHASES.md` are updated with verifiable test output.

