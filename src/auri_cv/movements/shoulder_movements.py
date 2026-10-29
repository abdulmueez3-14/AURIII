"""Movement-specific shoulder angle estimation and (separate) movement suggestion.

Each ShoulderMovement has its OWN definition: reference, plane/projection, sign, range,
required camera view and validity region. See docs/MOVEMENT_ANALYSIS.md.

All vectors are in the Torso Frame T (+X right, +Y anterior, +Z superior). u is the
shoulder->elbow unit vector; lateral components are sign-adjusted per arm so that
"lateral" is +x_lat for both the right and the left arm.

Honesty rules: a movement whose defining condition is not met returns angle None
(NOT_MEASURABLE), never a number borrowed from another movement. Confidence is a
HEURISTIC reliability indicator (not a probability, not an accuracy bound).
"""

from dataclasses import dataclass
from enum import StrEnum
from math import asin, atan2, degrees, hypot
from typing import Final

import numpy as np

from auri_cv.anatomy.segments import ArmLaterality, ArmSegmentState
from auri_cv.frames.torso_frame import TorsoFrame
from auri_cv.geometry.angles import signed_angle_around_axis
from auri_cv.geometry.vectors import normalize_vector, vector_norm


class ShoulderMovement(StrEnum):
    FLEXION = "FLEXION"
    EXTENSION = "EXTENSION"
    ABDUCTION = "ABDUCTION"
    ADDUCTION = "ADDUCTION"
    INTERNAL_ROTATION = "INTERNAL_ROTATION"
    EXTERNAL_ROTATION = "EXTERNAL_ROTATION"
    HORIZONTAL_ABDUCTION = "HORIZONTAL_ABDUCTION"
    HORIZONTAL_ADDUCTION = "HORIZONTAL_ADDUCTION"
    SCAPTION = "SCAPTION"


class CameraView(StrEnum):
    FRONTAL = "FRONTAL"  # subject faces (or has back to) the camera
    SAGITTAL = "SAGITTAL"  # subject side-on to the camera
    OBLIQUE = "OBLIQUE"
    UNKNOWN = "UNKNOWN"


class MotionStatus(StrEnum):
    OK = "OK"  # measurable, confidence >= OK_CONFIDENCE
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # angle given but must be labelled low confidence
    UNRELIABLE = "UNRELIABLE"  # confidence below the reporting floor: angle withheld
    NOT_MEASURABLE = "NOT_MEASURABLE"  # pose outside the movement's definition: no angle


HEURISTIC_NOTE: Final[str] = "confidence is a heuristic, not an accuracy bound"
OK_CONFIDENCE: Final[float] = 0.6
REPORT_FLOOR_CONFIDENCE: Final[float] = 0.25
MIN_ACTIVE_DEG: Final[float] = 20.0  # classifier: movement is "being performed" above this
OFF_PLANE_LIMIT_DEG: Final[float] = 25.0  # flexion/abduction: max angle out of own plane
SCAPTION_PLANE_DEG: Final[float] = 30.0  # scapular plane, anterior to the coronal plane
SCAPTION_TOLERANCE_DEG: Final[float] = 20.0
SCAPTION_PLANE_MIN_ELEV_DEG: Final[float] = 20.0  # plane of elevation unstable below this
HORIZONTAL_TOLERANCE_DEG: Final[float] = 25.0  # |elevation - 90| for horizontal movements
ROTATION_MAX_ELEVATION_DEG: Final[float] = 30.0  # rotation protocol: arm at the side
ROTATION_ELBOW_RANGE_DEG: Final[tuple[float, float]] = (70.0, 110.0)
ROTATION_MIN_FOREARM_PERP: Final[float] = 0.3
MIN_IN_PLANE: Final[float] = 0.0871  # sin(5 deg)


@dataclass(frozen=True)
class MovementSpec:
    movement: ShoulderMovement
    plane: str
    reference: str
    sign_convention: str
    nominal_range_deg: tuple[float, float]  # typical range in the positive direction
    preferred_view: CameraView
    acceptable_views: tuple[CameraView, ...]
    confidence_cap: float  # ceiling from monocular-depth sensitivity of this movement
    relevant_compensation: tuple[str, ...]
    requires: str


_F, _S, _O, _U = CameraView.FRONTAL, CameraView.SAGITTAL, CameraView.OBLIQUE, CameraView.UNKNOWN

SPECS: Final[dict[ShoulderMovement, MovementSpec]] = {
    ShoulderMovement.FLEXION: MovementSpec(
        ShoulderMovement.FLEXION, "sagittal (Y_T-Z_T)", "arm hanging (-Z_T) = 0",
        "+ = forward (anterior); angle = atan2(v_y, -v_z)", (0.0, 180.0), _S, (_O,), 0.9,
        ("forward trunk lean", "shoulder hiking"),
        "side view; arm within 25 deg of the sagittal plane",
    ),
    ShoulderMovement.EXTENSION: MovementSpec(
        ShoulderMovement.EXTENSION, "sagittal (Y_T-Z_T)", "arm hanging (-Z_T) = 0",
        "+ = backward (posterior); angle = -atan2(v_y, -v_z)", (0.0, 60.0), _S, (_O,), 0.9,
        ("forward trunk lean", "trunk rotation"),
        "side view; arm within 25 deg of the sagittal plane",
    ),
    ShoulderMovement.ABDUCTION: MovementSpec(
        ShoulderMovement.ABDUCTION, "coronal (X_T-Z_T)", "arm hanging (-Z_T) = 0",
        "+ = away from midline; angle = atan2(x_lat, -v_z)", (0.0, 180.0), _F, (_O,), 0.9,
        ("lateral trunk lean", "shoulder hiking"),
        "front/back view; arm within 25 deg of the coronal plane",
    ),
    ShoulderMovement.ADDUCTION: MovementSpec(
        ShoulderMovement.ADDUCTION, "coronal (X_T-Z_T)", "arm hanging (-Z_T) = 0",
        "+ = across the midline; angle = -atan2(x_lat, -v_z)", (0.0, 30.0), _F, (_O,), 0.9,
        ("lateral trunk lean", "trunk rotation"),
        "front/back view; arm within 25 deg of the coronal plane",
    ),
    ShoulderMovement.SCAPTION: MovementSpec(
        ShoulderMovement.SCAPTION, "scapular plane (30 deg anterior to coronal)",
        "arm hanging (-Z_T) = 0", "+ = elevation; angle = arccos(-v_z)", (0.0, 180.0), _O,
        (_F,), 0.85, ("lateral trunk lean", "shoulder hiking"),
        "oblique (30-60 deg) or frontal view; arm within 20 deg of the 30 deg plane",
    ),
    ShoulderMovement.HORIZONTAL_ADDUCTION: MovementSpec(
        ShoulderMovement.HORIZONTAL_ADDUCTION, "transverse (X_T-Y_T), arm at ~90 deg elevation",
        "arm abducted to 90 deg (lateral) = 0",
        "+ = toward/across the front; angle = atan2(v_y, x_lat)", (0.0, 130.0), _F, (_O,), 0.6,
        ("trunk rotation", "shoulder hiking"),
        "arm within 25 deg of horizontal; frontal view (anterior motion is depth: low confidence)",
    ),
    ShoulderMovement.HORIZONTAL_ABDUCTION: MovementSpec(
        ShoulderMovement.HORIZONTAL_ABDUCTION, "transverse (X_T-Y_T), arm at ~90 deg elevation",
        "arm abducted to 90 deg (lateral) = 0",
        "+ = backward from the coronal plane; angle = -atan2(v_y, x_lat)", (0.0, 45.0), _F,
        (_O,), 0.6, ("trunk rotation", "shoulder hiking"),
        "arm within 25 deg of horizontal; frontal view (posterior motion is depth: low confidence)",
    ),
    ShoulderMovement.EXTERNAL_ROTATION: MovementSpec(
        ShoulderMovement.EXTERNAL_ROTATION, "rotation about the upper-arm axis",
        "forearm pointing anterior, elbow ~90 deg, arm at side = 0",
        "+ = forearm swings outward (away from midline)", (0.0, 90.0), _F, (_O,), 0.5,
        ("trunk rotation", "shoulder hiking"),
        "elbow flexed 70-110 deg, arm within 30 deg of vertical, wrist visible; EXPERIMENTAL",
    ),
    ShoulderMovement.INTERNAL_ROTATION: MovementSpec(
        ShoulderMovement.INTERNAL_ROTATION, "rotation about the upper-arm axis",
        "forearm pointing anterior, elbow ~90 deg, arm at side = 0",
        "+ = forearm swings inward (toward/across midline)", (0.0, 70.0), _F, (_O,), 0.5,
        ("trunk rotation", "shoulder hiking"),
        "elbow flexed 70-110 deg, arm within 30 deg of vertical, wrist visible; EXPERIMENTAL",
    ),
}  # fmt: skip

VIEW_FACTOR_PREFERRED: Final[float] = 1.0
VIEW_FACTOR_ACCEPTABLE: Final[float] = 0.7
VIEW_FACTOR_POOR: Final[float] = 0.4
VIEW_FACTOR_UNKNOWN: Final[float] = 0.6


@dataclass(frozen=True)
class CameraAxes:
    """Camera lateral (image-x) and depth (optical-axis) directions in the WORLD frame."""

    lateral: tuple[float, float, float] = (1.0, 0.0, 0.0)
    depth: tuple[float, float, float] = (0.0, 0.0, 1.0)


MEDIAPIPE_CAMERA: Final[CameraAxes] = CameraAxes()  # MediaPipe world: x image-right, z depth


def estimate_subject_yaw_deg(frame: TorsoFrame, camera: CameraAxes = MEDIAPIPE_CAMERA) -> float:
    """Yaw of the shoulder line vs. the image plane: 0 = frontal, 90 = side-on.

    Folded to [0, 90]: front/back facing and left/right side-on are indistinguishable.
    Assumes world axes are camera-aligned (true for MediaPipe world landmarks).
    """
    lat = abs(float(np.dot(frame.x_axis, normalize_vector(np.array(camera.lateral)))))
    dep = abs(float(np.dot(frame.x_axis, normalize_vector(np.array(camera.depth)))))
    return degrees(atan2(dep, lat))


def classify_view(yaw_deg: float | None) -> CameraView:
    if yaw_deg is None:
        return CameraView.UNKNOWN
    if yaw_deg <= 25.0:
        return CameraView.FRONTAL
    if yaw_deg >= 65.0:
        return CameraView.SAGITTAL
    return CameraView.OBLIQUE


@dataclass(frozen=True)
class MotionEstimate:
    """Output contract for one movement-specific measurement (degrees at the boundary)."""

    movement: ShoulderMovement
    laterality: ArmLaterality
    angle_deg: float | None  # None unless status is OK or LOW_CONFIDENCE
    status: MotionStatus
    confidence: float  # heuristic reliability in [0, 1]; 0 when angle is None
    reasons: tuple[str, ...]  # why confidence is reduced / why not measurable
    method: str
    reference: str
    view: CameraView
    view_yaw_deg: float | None
    elevation_deg: float  # total humerothoracic elevation (context, always defined)
    plane_of_elevation_deg: float | None
    within_nominal_range: bool | None
    timestamp_ms: float


@dataclass(frozen=True)
class _Geometry:
    angle_deg: float | None
    off_plane_deg: float
    off_plane_limit_deg: float
    reason: str | None  # set when not measurable
    method: str


def _lateral_sign(lat: ArmLaterality) -> float:
    return 1.0 if lat == ArmLaterality.RIGHT else -1.0


def _geometry(movement: ShoulderMovement, seg: ArmSegmentState) -> _Geometry:
    u = normalize_vector(seg.upper_arm_dir_torso)
    x_lat = _lateral_sign(seg.laterality) * float(u[0])
    vy, vz = float(u[1]), float(u[2])
    elev = degrees(float(np.arccos(np.clip(-vz, -1.0, 1.0))))
    method = SPECS[movement].sign_convention

    def fail(why: str) -> _Geometry:
        return _Geometry(None, 0.0, 1.0, why, method)

    if movement in (ShoulderMovement.FLEXION, ShoulderMovement.EXTENSION):
        if hypot(vy, vz) < MIN_IN_PLANE:
            return fail("arm has no sagittal-plane component")
        off = degrees(asin(min(1.0, abs(x_lat))))
        if off > OFF_PLANE_LIMIT_DEG:
            return fail(f"arm is {off:.0f} deg out of the sagittal plane (not a flexion/extension)")
        a = degrees(atan2(vy, -vz))
        return _Geometry(a if movement == ShoulderMovement.FLEXION else -a, off,
                         OFF_PLANE_LIMIT_DEG, None, method)  # fmt: skip

    if movement in (ShoulderMovement.ABDUCTION, ShoulderMovement.ADDUCTION):
        if hypot(x_lat, vz) < MIN_IN_PLANE:
            return fail("arm has no coronal-plane component")
        off = degrees(asin(min(1.0, abs(vy))))
        if off > OFF_PLANE_LIMIT_DEG:
            return fail(f"arm is {off:.0f} deg out of the coronal plane (not abduction/adduction)")
        a = degrees(atan2(x_lat, -vz))
        return _Geometry(a if movement == ShoulderMovement.ABDUCTION else -a, off,
                         OFF_PLANE_LIMIT_DEG, None, method)  # fmt: skip

    if movement == ShoulderMovement.SCAPTION:
        off = 0.0
        if elev >= SCAPTION_PLANE_MIN_ELEV_DEG:
            dev = degrees(atan2(vy, x_lat)) - SCAPTION_PLANE_DEG
            dev = (dev + 180.0) % 360.0 - 180.0
            off = abs(dev)
            if off > SCAPTION_TOLERANCE_DEG:
                return fail(f"arm plane is {off:.0f} deg from the scapular plane")
        return _Geometry(elev, off, SCAPTION_TOLERANCE_DEG, None, method)

    if movement in (ShoulderMovement.HORIZONTAL_ABDUCTION, ShoulderMovement.HORIZONTAL_ADDUCTION):
        off = abs(elev - 90.0)
        if off > HORIZONTAL_TOLERANCE_DEG:
            return fail(f"arm elevation {elev:.0f} deg: horizontal movement needs ~90 deg")
        psi = degrees(atan2(vy, x_lat))
        a = psi if movement == ShoulderMovement.HORIZONTAL_ADDUCTION else -psi
        return _Geometry(a, off, HORIZONTAL_TOLERANCE_DEG, None, method)

    # Axial rotation: only under the elbow-90 / arm-at-side protocol (ADR-006, ADR-011).
    if seg.forearm_dir_torso is None or seg.elbow_flexion_deg is None:
        return fail("forearm/wrist not visible: axial rotation is unobservable")
    lo, hi = ROTATION_ELBOW_RANGE_DEG
    if not lo <= seg.elbow_flexion_deg <= hi:
        return fail(
            f"elbow flexion {seg.elbow_flexion_deg:.0f} deg outside protocol {lo:.0f}-{hi:.0f}"
        )
    if elev > ROTATION_MAX_ELEVATION_DEG:
        return fail(f"arm elevation {elev:.0f} deg: rotation protocol needs the arm at the side")
    f = normalize_vector(seg.forearm_dir_torso)
    f_perp = f - float(np.dot(f, u)) * u
    if vector_norm(f_perp) < ROTATION_MIN_FOREARM_PERP:
        return fail("forearm is aligned with the upper arm: rotation unobservable")
    anterior = np.array([0.0, 1.0, 0.0])
    er = signed_angle_around_axis(anterior, f_perp, u, in_degrees=True)  # right arm: + = outward
    er *= _lateral_sign(seg.laterality)
    a = er if movement == ShoulderMovement.EXTERNAL_ROTATION else -er
    return _Geometry(a, elev, ROTATION_MAX_ELEVATION_DEG, None, method)


def _view_factor(spec: MovementSpec, view: CameraView) -> tuple[float, str | None]:
    if view == CameraView.UNKNOWN:
        return VIEW_FACTOR_UNKNOWN, "camera view unknown"
    if view == spec.preferred_view:
        return VIEW_FACTOR_PREFERRED, None
    if view in spec.acceptable_views:
        return (
            VIEW_FACTOR_ACCEPTABLE,
            f"{view.value} view acceptable, {spec.preferred_view.value} preferred",
        )
    return (
        VIEW_FACTOR_POOR,
        f"{view.value} view is poor for this movement ({spec.preferred_view.value} needed)",
    )  # noqa: E501


def estimate_motion(
    movement: ShoulderMovement,
    segment: ArmSegmentState,
    *,
    quality_score: float = 1.0,
    camera_yaw_deg: float | None = None,
    timestamp_ms: float = 0.0,
) -> MotionEstimate:
    """Movement-specific angle for the SELECTED movement (selection is explicit)."""
    spec = SPECS[movement]
    u = normalize_vector(segment.upper_arm_dir_torso)
    x_lat = _lateral_sign(segment.laterality) * float(u[0])
    elev = degrees(float(np.arccos(np.clip(-float(u[2]), -1.0, 1.0))))
    plane = (
        degrees(atan2(float(u[1]), x_lat)) if hypot(x_lat, float(u[1])) >= MIN_IN_PLANE else None
    )
    view = classify_view(camera_yaw_deg)
    geo = _geometry(movement, segment)

    def make(angle, status, conf, reasons, in_range=None):  # noqa: ANN001, ANN202
        return MotionEstimate(
            movement, segment.laterality, angle, status, conf, tuple(reasons), geo.method,
            spec.reference, view, camera_yaw_deg, elev, plane, in_range, timestamp_ms,
        )  # fmt: skip

    if geo.angle_deg is None:
        return make(None, MotionStatus.NOT_MEASURABLE, 0.0, [geo.reason or "not measurable"])

    reasons: list[str] = []
    vf, vreason = _view_factor(spec, view)
    if vreason:
        reasons.append(vreason)
    plane_factor = 1.0 - 0.5 * min(1.0, geo.off_plane_deg / geo.off_plane_limit_deg)
    if geo.off_plane_deg > 0.5 * geo.off_plane_limit_deg:
        reasons.append(f"arm {geo.off_plane_deg:.0f} deg from the movement's defining plane/pose")
    if spec.confidence_cap < 0.9:
        reasons.append(f"depth-sensitive movement: confidence capped at {spec.confidence_cap}")
    conf = float(min(spec.confidence_cap, max(0.0, min(1.0, quality_score)) * vf * plane_factor))

    lo, hi = spec.nominal_range_deg
    in_range = lo <= geo.angle_deg <= hi
    if geo.angle_deg < lo:
        reasons.append("negative angle: arm is in the antagonist direction")
    elif geo.angle_deg > hi:
        reasons.append("beyond typical range: check tracking")

    if conf < REPORT_FLOOR_CONFIDENCE:
        return make(
            None, MotionStatus.UNRELIABLE, conf, ["below reporting floor", *reasons, HEURISTIC_NOTE]
        )
    status = MotionStatus.OK if conf >= OK_CONFIDENCE else MotionStatus.LOW_CONFIDENCE
    return make(geo.angle_deg, status, conf, [*reasons, HEURISTIC_NOTE], in_range)


@dataclass(frozen=True)
class MovementSuggestion:
    """Optional hint, NEVER a substitute for explicit exercise selection."""

    movement: ShoulderMovement | None  # set only when exactly one candidate fits
    candidates: tuple[ShoulderMovement, ...]
    note: str


_CLASSIFIABLE: Final[tuple[ShoulderMovement, ...]] = tuple(
    m
    for m in ShoulderMovement
    if m not in (ShoulderMovement.INTERNAL_ROTATION, ShoulderMovement.EXTERNAL_ROTATION)
)


def suggest_movement(segment: ArmSegmentState) -> MovementSuggestion:
    """Which movements' definitions the current pose satisfies (rotation is never inferred).

    Several poses are intrinsically ambiguous (e.g. arm forward at 90 deg is both flexion and
    horizontal adduction), so more than one candidate is returned and `movement` stays None.
    """
    cands = tuple(
        m
        for m in _CLASSIFIABLE
        if (g := _geometry(m, segment)).angle_deg is not None and g.angle_deg > MIN_ACTIVE_DEG
    )
    if not cands:
        return MovementSuggestion(None, (), "near neutral or outside every movement definition")
    if len(cands) == 1:
        return MovementSuggestion(cands[0], cands, "single matching definition")
    return MovementSuggestion(None, cands, "ambiguous pose: select the exercise explicitly")
