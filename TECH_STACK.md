# TECH_STACK.md — Technology Stack and Dependency Register
## AURI: Computer Vision & Biomedical Engine

---

## 1. Core Technology Register

| Technology | Purpose | Selected Version | Integration Status | Selection Rationale | Alternatives Evaluated & Rejected | Key Technical Risks |
|---|---|---|---|---|---|---|
| **Python** | Primary programming language | `3.12.10` | **Verified & Installed** | Long-Term Support release with high ecosystem stability, mature C-extension wheels (NumPy, SciPy, OpenCV, MediaPipe), and stable typing syntax. | Python 3.14 (experimental, pre-release C-extension wheels lacking for many scientific packages), Python 3.10/3.11 (older LTS). | Future deprecation in late 2028; easily managed via virtual environment. |
| **MediaPipe Tasks** | Pretrained monocular 2D & estimated 3D human pose estimation | `1.2.0` (`PoseLandmarker`) | **Verified & Installed** | Google's official, actively maintained Tasks Vision API. Provides 33 2D normalized landmarks and 33 metric 3D world landmarks with presence/visibility scores. | Legacy `mediapipe.solutions.pose` (deprecated and removed in MediaPipe 1.x), OpenPose (heavyweight C++ dependency, non-commercial license), YOLO-Pose (2D only, no metric world landmarks). | Z-depth is model-inferred, not physical depth; susceptible to foreshortening and monocular scale ambiguity. |
| **NumPy** | Vector, matrix, and array computations | `2.5.4` | **Verified & Installed** | Standard high-performance vectorized linear algebra engine in Python; fast C-accelerated array operations. | Pure Python `math` (too slow for real-time video), JAX/PyTorch (unnecessary overhead for low-dimensional 3D kinematics). | NumPy 2.0 API breaking changes for legacy C extensions (verified compatible with installed packages). |
| **SciPy** | Rotation representations and digital signal filtering | `1.18.1` | **Verified & Installed** | Battle-tested implementations of SO(3) rotations (`scipy.spatial.transform.Rotation`) and digital filters (Butterworth, Savitzky-Golay). | Writing custom quaternion/Euler solvers from scratch (error-prone, reinventing tested numerical mathematics). | Overhead if invoked in tight inner loops (mitigated by direct vector formulations for core operations). |
| **OpenCV** (`opencv-python`) | Video acquisition, frame extraction, image processing, diagnostic overlays | `5.0.0.93` | **Verified & Installed** | Industry-standard computer vision library for low-latency webcam capture, video decoding, coordinate drawing, and color conversions. | PyAV / imageio (more complex for real-time capture), GStreamer (heavy OS-level dependency). | Webcam backend differences across Windows (`CAP_DSHOW`, `CAP_MSMF`). |
| **Pydantic** | Typed structured observation data models and validation | `2.14.1` | **Verified & Installed** | Fast, Rust-backed schema validation ensuring rigorous data contracts between pipeline stages. | Standard `dataclasses` (lack runtime boundary validation), `NamedTuple` (inflexible). | Slight serialization overhead if applied per-landmark; mitigated by applying at frame/pipeline boundaries. |
| **pytest** | Unit, synthetic, and regression testing framework | `9.2.0` | **Verified & Installed** | Standard test runner with rich assertion introspection, parameterized tests, and fixture support. | `unittest` (verbose, less flexible test parametrization). | None. |
| **Ruff** | Code formatting and static linting | `0.17.3` | **Verified & Installed** | Extremely fast Rust-based linter and formatter replacing Flake8, Black, and isort. | Black + Flake8 + isort (slower, fragmented configuration). | None. |
| **Matplotlib** | Offline trajectory visualization and validation plotting | `3.11.2` | **Verified & Installed** | Generation of publication-quality kinematic curves, Bland-Altman plots, and validation comparisons. | Seaborn (additional layer), Plotly (unnecessary browser dependencies). | GUI backend rendering issues in headless environments (use `Agg` backend). |

---

## 2. Dependencies Strictly Out of Scope

| Component | Status | Rationale for Exclusion |
|---|---|---|
| **Flutter / Dart** | **EXCLUDED** | AURI CV is a standalone research engine. Mobile app shell belongs to future integration phase. |
| **Android Studio / Kotlin** | **EXCLUDED** | Premature coupling to mobile platforms; inhibits rapid mathematical experimentation and rigorous validation. |
| **BLE Libraries (`bleak`, etc.)** | **EXCLUDED** | Physical IMU hardware integration is decoupled until Phase 9. Hardware communication belongs in a dedicated device driver layer. |
| **Cloud Services / Firebase** | **EXCLUDED** | Privacy and latency require local-first, zero-cloud deterministic computation. |
| **Custom Deep Learning Training** | **EXCLUDED** | Using MediaPipe's validated pose model; training custom pose neural nets is unnecessary and unverified. |

---

## 3. Environment Specifications
- **Operating System**: Windows 11 (MSVC x86_64).
- **Python Executable**: `.venv/Scripts/python.exe` (CPython 3.12.10).
- **Environment Manager**: `uv` (v0.11.3) with localized `.venv`.
- **Packaging Standard**: PEP 621 (`pyproject.toml`).

