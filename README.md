# AURI: Computer Vision & Biomedical Engine
### Assisted Upper-Limb Rehabilitation Intelligence

A modular, explainable, and testable Python biomedical research engine for camera-based shoulder movement analysis, initially focused on adhesive capsulitis (frozen shoulder) rehabilitation.

---

## 1. Project Mission & Architecture

AURI transforms camera observations into verified, geometrically meaningful shoulder kinematics. It explicitly distinguishes between:
1. **Video Acquisition** (Webcam / Video file ingestion).
2. **Pose Estimation** (MediaPipe Tasks Vision `PoseLandmarker`).
3. **Landmark Normalization & Quality Assessment**.
4. **Torso Coordinate-Frame Estimation** (Subject-centric orthonormal frame).
5. **Anatomical Segment Estimation** ($\mathbb{S}^2$ unit vectors).
6. **Joint Kinematics & Angle Solving** (Elevation, sagittal flexion, coronal abduction).
7. **Reference-Pose Calibration & Neutral-Zone Zeroing** (Rock-solid $0.0^\circ$ display with hysteresis).
8. **Movement Analysis & Repetition Detection** (Finite State Machine).
9. **Compensation Detection** (Trunk lean, rotation, shoulder hiking).
10. **Measurement Quality Assessment** (`VALID`, `VALID_WITH_WARNING`, `UNAVAILABLE`, `RECALIBRATION_REQUIRED`).

```text
Video → Pose Estimator → Landmark Normalizer → Landmark Quality Gate 
      → Torso-Frame Estimator → Segment Estimator → Joint-Angle Solver 
      → Calibration & Neutral Zone → Movement Analyzer → Measurement Quality Gate → Results
```

---

## 2. Clinical Boundaries & Disclaimer

> [!CAUTION]
> **Rehabilitation Support Only**:
> AURI is an engineering and biomedical research system. It does **not** provide autonomous diagnosis, disease staging, or medical prescription. It measures total **humerothoracic elevation** and must not be conflated with isolated glenohumeral range of motion.

---

## 3. Technology Stack & Prerequisites

- **Python**: 3.12 LTS
- **Core Dependencies**: `mediapipe>=1.2.0`, `opencv-python>=4.10.0`, `numpy>=2.0.0`, `scipy>=1.14.0`, `pydantic>=2.10.0`
- **Testing & Tools**: `pytest>=8.0.0`, `ruff>=0.9.0`, `matplotlib>=3.10.0`
- **Environment Manager**: `uv` or standard Python `venv`

---

## 4. Quick Start

### 4.1 Setup Virtual Environment
```bash
# Using uv (recommended)
uv venv .venv --python 3.12
.\.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 4.2 Run Test Suite
```bash
pytest
```

### 4.3 Run Linter & Formatter
```bash
ruff check .
ruff format --check .
```

---

## 5. Repository Documentation Structure

- [AGENTS.md](AGENTS.md): Persistent coding-agent operating rules and guardrails.
- [PRD.md](PRD.md): Product Requirements Document with traceable IDs.
- [TECH_STACK.md](TECH_STACK.md): Full dependency register, rationales, and alternatives.
- [PHASES.md](PHASES.md): Implementation roadmap and milestone criteria.
- [PROJECT_LOG.md](PROJECT_LOG.md): Canonical state summary and iteration log.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): Detailed pipeline architecture and contracts.
- [docs/MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md): Equations, derivations, safeguards, and worked examples.
- [docs/BIOMECHANICAL_CONVENTIONS.md](docs/BIOMECHANICAL_CONVENTIONS.md): Coordinate frames, handedness, and units.
- [docs/VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md): 6-tier verification and synthetic test definitions.
- [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md): Observability boundaries and depth uncertainties.
- [docs/DECISIONS.md](docs/DECISIONS.md): Architectural Decision Records (ADRs).

