# Biomechanical Conventions and Coordinate Standards
## AURI: Computer Vision & Biomedical Engine

---

## 1. Coordinate Frames of Reference

The AURI architecture recognizes and formally distinguishes between five spatial reference frames:

```mermaid
flowchart LR
    IMG["2D Image Frame (u, v)"] -. Projection .-> CAM["3D Camera Frame (C)"]
    CAM -. Model Inference .-> MP["MediaPipe World Frame (W)"]
    MP --> T["Torso Coordinate Frame (T)"]
    T --> S["Segment Frames (A)"]
```

### 1.1 Normalized Image Frame ($I_{2D}$)
- **Origin**: Top-left corner of the video frame $(0.0, 0.0)$.
- **Axes**:
  - $+u$: Horizontal axis pointing from image left to image right, normalized to $[0.0, 1.0]$.
  - $+v$: Vertical axis pointing from image top to image bottom, normalized to $[0.0, 1.0]$.
- **Usage**: Diagnostic video overlays, landmark presence and screen-boundary bounding checks. Never used for 3D anatomical angle computation.

### 1.2 MediaPipe 3D World Landmark Frame ($W$)
- **Origin**: Hip center landmark midpoint at mid-pelvis.
- **Axes**:
  - $+X_W$: Points to the subject's anatomical right (screen left in frontal selfie view).
  - $+Y_W$: Points vertically downward (towards the floor).
  - $+Z_W$: Points forward toward or away from the camera depending on perspective; treated as depth relative to hips in metric units.
- **Units**: Approximate meters.
- **Warning**: World landmarks are model-inferred metric approximations derived from monocular image features, not physical ground-truth LiDAR/optical motion capture points.

### 1.3 Subject-Centered Torso Frame ($T$)
- **Origin ($\mathbf{o}_T$)**: Shoulder center midpoint $\mathbf{S}_c = \frac{\mathbf{S}_L + \mathbf{S}_R}{2}$.
- **Axes**:
  - $+X_T$: Lateral axis pointing to the subject's anatomical right.
  - $+Y_T$: Anterior axis pointing straight out of the chest (ventral direction).
  - $+Z_T$: Superior axis pointing headward along the spinal column.
- **Handedness**: Right-handed Cartesian system ($\hat{\mathbf{x}}_T \times \hat{\mathbf{y}}_T = \hat{\mathbf{z}}_T$).
- **Properties**: Rigidly attached to the subject's trunk; decouples arm movement from gross trunk lean, sway, or axial twisting.

### 1.4 Segment Direction Frame ($A$)
- Upper arm segment direction $\hat{\mathbf{v}}_{\text{arm}}$ (proximal $\mathbf{S}$ to distal $\mathbf{E}$).
- Forearm segment direction $\hat{\mathbf{v}}_{\text{forearm}}$ (proximal $\mathbf{E}$ to distal $\mathbf{W}$).
- Segments are treated as 1D spatial vectors on the unit sphere ($\mathbb{S}^2$). Full SO(3) orientations require future IMU sensor integration.

---

## 2. Units and Numerical Standards

| Physical Quantity | Internal Pipeline Unit | Display / Output Boundary Unit |
|---|---|---|
| Spatial Coordinates | Meters ($\text{m}$) | Centimeters ($\text{cm}$) or Meters ($\text{m}$) |
| Angles | Radians ($\text{rad}$) | Degrees ($^\circ$) |
| Angular Velocity | Radians / Second ($\text{rad/s}$) | Degrees / Second ($^\circ/\text{s}$) |
| Time / Timestamps | Seconds ($\text{s}$) or Milliseconds ($\text{ms}$) | Seconds ($\text{s}$) / Integer Milliseconds |
| Frequency | Hertz ($\text{Hz}$) | Frames Per Second ($\text{FPS}$) |

---

## 3. Rotation Matrix and Quaternion Standards

### 3.1 Rotation Matrix Convention
- Rotation matrices are $3 \times 3$ real orthogonal matrices belonging to the Special Orthogonal group $\text{SO}(3)$:
  $$R^T R = I_{3 \times 3}, \quad \det(R) = +1.0$$
- **Transformation Direction Notation**:
  $R_{B \leftarrow A}$ (or `R_A_to_B`) transforms the coordinates of a vector expressed in Frame $A$ into Frame $B$:
  $$\mathbf{v}_B = R_{B \leftarrow A} \mathbf{v}_A$$
- In column-vector notation, the columns of $R_{B \leftarrow A}$ are the basis unit vectors of Frame $A$ expressed in Frame $B$:
  $$R_{B \leftarrow A} = \begin{bmatrix} \hat{\mathbf{x}}_A^{(B)} & \hat{\mathbf{y}}_A^{(B)} & \hat{\mathbf{z}}_A^{(B)} \end{bmatrix}$$

### 3.2 Quaternion Standards
Where quaternions are utilized (e.g., in future IMU contracts or SciPy conversions):
- **Ordering**: Hamilton scalar-first convention:
  $$\mathbf{q} = (w, x, y, z) = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$$
- **Unit Norm Constraint**: $\|\mathbf{q}\| = \sqrt{w^2 + x^2 + y^2 + z^2} = 1.0$.

---

## 4. Anatomical Neutral vs. Functional Starting Posture

1. **True Anatomical Neutral**:
   - Standing erect, feet together, head facing forward.
   - Arms hanging at sides, palms facing anteriorly (supinated).
   - Biomechanical glenohumeral angle is strictly $0.0^\circ$.
2. **Clinical Functional Reference Posture**:
   - Patient seated or standing comfortably facing the camera.
   - Arms resting naturally at sides, palms resting against outer thighs (mid-pronation).
   - In patients with adhesive capsulitis, resting posture often exhibits mild resting abduction (e.g. 5°–15°) or internal rotation due to capsular contracture.
3. **AURI's Resolution**:
   - The engine measures **relative excursion** $\Delta\theta(t) = \theta(t) - \theta_0$ from the patient's individual calibrated functional reference posture $\theta_0$.
   - $\theta_0$ is explicitly retained in the patient's session record. It is never conflated with true anatomical zero.

---

## 5. Laterality and Mirroring Rules

### 5.1 Left vs. Right Shoulder Laterality
Because $+X_T$ points to the subject's anatomical right:
- **Right Arm Abduction**: Moving the right arm laterally points along $+X_T$.
  $$\theta_{\text{abduction, R}} = \operatorname{atan2}(v_x, -v_z)$$
- **Left Arm Abduction**: Moving the left arm laterally points along $-X_T$.
  $$\theta_{\text{abduction, L}} = \operatorname{atan2}(-v_x, -v_z)$$
- **Forward Flexion**: For both arms, moving anteriorly points along $+Y_T$:
  $$\theta_{\text{flexion}} = \operatorname{atan2}(v_y, -v_z)$$

### 5.2 Video Mirroring
- Standard webcams in video call / selfie mode mirror the horizontal axis ($x \mapsto 1 - x$).
- MediaPipe internally tracks true anatomical left and right landmarks regardless of mirroring if correctly configured.
- AURI's video ingestion layer explicitly tracks `is_mirrored: bool` to ensure overlays and lateral directions match clinical expectation.

---

## 6. Humerothoracic vs. Glenohumeral Distinction

> [!WARNING]
> Surface optical landmarks (shoulder, elbow, wrist) capture **total humerothoracic elevation** — the combined motion of the glenohumeral joint (GH) and the scapulothoracic articulation (ST) on the rib cage (scapulohumeral rhythm, typically 2:1 ratio).
> AURI's camera measurements represent **humerothoracic elevation**. They must never be documented or presented to clinicians as isolated glenohumeral range of motion.

