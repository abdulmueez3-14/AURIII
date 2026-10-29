"""Independent synthetic tests for every movement-specific estimator (analytic geometry)."""

import numpy as np
import pytest

from auri_cv.anatomy.segments import ArmLaterality, ArmSegmentState
from auri_cv.frames.torso_frame import estimate_torso_frame
from auri_cv.movements import (
    CameraAxes,
    CameraView,
    CircumductionTracker,
    MotionStatus,
    ShoulderMovement,
    classify_view,
    estimate_motion,
    estimate_subject_yaw_deg,
    suggest_movement,
)
from tests.synthetic.fixtures import make_neutral_standing_pose

M = ShoulderMovement
R, L = ArmLaterality.RIGHT, ArmLaterality.LEFT
SAG, FRO = 90.0, 0.0  # camera yaw of the subject


def _u(elev: float, plane: float, lat: ArmLaterality = R) -> np.ndarray:
    """Torso-frame upper-arm direction; plane 0 = lateral, 90 = anterior, -90 = posterior."""
    e, p = np.radians(elev), np.radians(plane)
    sign = 1.0 if lat == R else -1.0
    return np.array([sign * np.sin(e) * np.cos(p), np.sin(e) * np.sin(p), -np.cos(e)])


def _forearm_rot(er_deg: float, lat: ArmLaterality = R) -> np.ndarray:
    """Forearm at 90 deg elbow flexion, arm hanging; +er = swung outward."""
    t = np.radians(er_deg)
    sign = 1.0 if lat == R else -1.0
    return np.array([sign * np.sin(t), np.cos(t), 0.0])


def _seg(u, lat: ArmLaterality = R, forearm=None) -> ArmSegmentState:
    u = np.asarray(u, dtype=float)
    f = None if forearm is None else np.asarray(forearm, dtype=float)
    elbow = None if f is None else float(np.degrees(np.arccos(np.clip(u @ f, -1, 1))))
    return ArmSegmentState(lat, u, u, 0.3, f, f, None if f is None else 0.25, elbow)


def _angle(movement, u, lat=R, forearm=None, yaw=None, quality=1.0):
    return estimate_motion(
        movement, _seg(u, lat, forearm), quality_score=quality, camera_yaw_deg=yaw
    )


# ---------------------------------------------------------------- per-movement correctness
@pytest.mark.parametrize("lat", [R, L])
@pytest.mark.parametrize("elev", [0.0, 30.0, 90.0, 150.0, 180.0])
def test_flexion(elev, lat):
    est = _angle(M.FLEXION, _u(elev, 90.0, lat), lat, yaw=SAG)
    assert est.angle_deg == pytest.approx(elev, abs=1e-6)
    assert est.status == MotionStatus.OK


@pytest.mark.parametrize("lat", [R, L])
@pytest.mark.parametrize("back", [10.0, 30.0, 55.0])
def test_extension_is_positive_backward_and_flexion_goes_negative(back, lat):
    u = _u(back, -90.0, lat)
    assert _angle(M.EXTENSION, u, lat, yaw=SAG).angle_deg == pytest.approx(back, abs=1e-6)
    flex = _angle(M.FLEXION, u, lat, yaw=SAG)
    assert flex.angle_deg == pytest.approx(-back, abs=1e-6)
    assert flex.within_nominal_range is False
    assert any("antagonist" in r for r in flex.reasons)


@pytest.mark.parametrize("lat", [R, L])
@pytest.mark.parametrize("elev", [0.0, 45.0, 90.0, 135.0, 180.0])
def test_abduction(elev, lat):
    est = _angle(M.ABDUCTION, _u(elev, 0.0, lat), lat, yaw=FRO)
    assert est.angle_deg == pytest.approx(elev, abs=1e-6) and est.status == MotionStatus.OK


@pytest.mark.parametrize("lat", [R, L])
def test_adduction_is_positive_across_the_midline(lat):
    u = _u(20.0, 180.0, lat)  # arm moved across the body
    assert _angle(M.ADDUCTION, u, lat, yaw=FRO).angle_deg == pytest.approx(20.0, abs=1e-6)
    assert _angle(M.ABDUCTION, u, lat, yaw=FRO).angle_deg == pytest.approx(-20.0, abs=1e-6)


@pytest.mark.parametrize("lat", [R, L])
@pytest.mark.parametrize("elev", [40.0, 90.0, 120.0])
def test_scaption_in_scapular_plane(elev, lat):
    est = _angle(M.SCAPTION, _u(elev, 30.0, lat), lat, yaw=45.0)
    assert est.angle_deg == pytest.approx(elev, abs=1e-6) and est.status == MotionStatus.OK


@pytest.mark.parametrize("lat", [R, L])
@pytest.mark.parametrize("psi", [0.0, 30.0, 90.0, 130.0])
def test_horizontal_adduction(psi, lat):
    est = _angle(M.HORIZONTAL_ADDUCTION, _u(90.0, psi, lat), lat, yaw=FRO)
    assert est.angle_deg == pytest.approx(psi, abs=1e-6)


@pytest.mark.parametrize("lat", [R, L])
@pytest.mark.parametrize("psi", [-10.0, -30.0, -45.0])
def test_horizontal_abduction_positive_backward(psi, lat):
    est = _angle(M.HORIZONTAL_ABDUCTION, _u(90.0, psi, lat), lat, yaw=FRO)
    assert est.angle_deg == pytest.approx(-psi, abs=1e-6)
    adduction = _angle(M.HORIZONTAL_ADDUCTION, _u(90.0, psi, lat), lat, yaw=FRO)
    assert adduction.angle_deg == pytest.approx(psi, abs=1e-6)  # negative = antagonist


@pytest.mark.parametrize("lat", [R, L])
@pytest.mark.parametrize("er", [0.0, 20.0, 45.0, 80.0])
def test_external_and_internal_rotation(er, lat):
    u = _u(0.0, 0.0, lat)
    f = _forearm_rot(er, lat)
    ext = _angle(M.EXTERNAL_ROTATION, u, lat, f, yaw=FRO)
    assert ext.angle_deg == pytest.approx(er, abs=1e-6)
    ir = _angle(M.INTERNAL_ROTATION, u, lat, _forearm_rot(-er, lat), yaw=FRO)
    assert ir.angle_deg == pytest.approx(er, abs=1e-6)
    # opposite sign when the forearm is in the antagonist direction
    assert _angle(M.INTERNAL_ROTATION, u, lat, f, yaw=FRO).angle_deg == pytest.approx(-er)


def test_rotation_never_exceeds_low_confidence():
    est = _angle(M.EXTERNAL_ROTATION, _u(0, 0), R, _forearm_rot(30.0), yaw=FRO)
    assert est.status == MotionStatus.LOW_CONFIDENCE and est.confidence <= 0.5


# ---------------------------------------------------------------- neutral & sweeps
@pytest.mark.parametrize("m", [m for m in M if "ROTATION" not in m and "HORIZONTAL" not in m])
def test_neutral_is_zero_for_planar_movements(m):
    est = _angle(m, _u(0.0, 0.0), yaw=45.0)
    assert est.angle_deg == pytest.approx(0.0, abs=1e-6)


def test_increasing_then_decreasing_sweep_is_monotonic():
    up = [_angle(M.ABDUCTION, _u(e, 0.0), yaw=FRO).angle_deg for e in range(0, 181, 15)]
    assert up == sorted(up) and up[-1] == pytest.approx(180.0)
    down = up[::-1]
    assert down == sorted(down, reverse=True)


# ---------------------------------------------------------------- confusion guards
def test_flexion_is_not_confused_with_abduction():
    abducted = _u(90.0, 0.0)
    assert _angle(M.FLEXION, abducted, yaw=SAG).status == MotionStatus.NOT_MEASURABLE
    assert _angle(M.FLEXION, abducted, yaw=SAG).angle_deg is None
    flexed = _u(90.0, 90.0)
    assert _angle(M.ABDUCTION, flexed, yaw=FRO).status == MotionStatus.NOT_MEASURABLE
    scaption_arm = _u(90.0, 30.0)
    assert _angle(M.FLEXION, scaption_arm, yaw=SAG).angle_deg is None
    assert _angle(M.ABDUCTION, scaption_arm, yaw=FRO).angle_deg is None


def test_scaption_rejects_pure_abduction_and_pure_flexion_planes():
    assert _angle(M.SCAPTION, _u(90.0, 0.0), yaw=45).status == MotionStatus.NOT_MEASURABLE
    assert _angle(M.SCAPTION, _u(90.0, 90.0), yaw=45).status == MotionStatus.NOT_MEASURABLE


def test_horizontal_requires_arm_near_horizontal():
    assert _angle(M.HORIZONTAL_ADDUCTION, _u(0.0, 0.0), yaw=FRO).angle_deg is None
    assert _angle(M.HORIZONTAL_ABDUCTION, _u(160.0, 0.0), yaw=FRO).angle_deg is None


def test_rotation_is_not_confused_with_elevation_and_vice_versa():
    # Rotating the forearm with the arm at the side leaves every elevation metric at ~0.
    u = _u(0.0, 0.0)
    for er in (0.0, 40.0, -40.0):
        e = _angle(M.EXTERNAL_ROTATION, u, R, _forearm_rot(er), yaw=FRO)
        assert e.elevation_deg == pytest.approx(0.0, abs=1e-6)
        assert _angle(M.ABDUCTION, u, R, _forearm_rot(er), yaw=FRO).angle_deg == pytest.approx(0.0)
        assert _angle(M.FLEXION, u, R, _forearm_rot(er), yaw=SAG).angle_deg == pytest.approx(0.0)
    # An elevated arm is not an axial rotation, whatever the forearm does.
    raised = _u(90.0, 0.0)
    assert _angle(M.EXTERNAL_ROTATION, raised, R, _forearm_rot(30.0), yaw=FRO).angle_deg is None


def test_elbow_motion_is_not_shoulder_motion():
    u = _u(60.0, 90.0)
    straight = _angle(M.FLEXION, u, R, u, yaw=SAG)  # forearm along upper arm
    bent = _angle(M.FLEXION, u, R, _forearm_rot(0.0), yaw=SAG)  # elbow bent, same upper arm
    assert straight.angle_deg == pytest.approx(bent.angle_deg)


def test_rotation_protocol_gates():
    u = _u(0.0, 0.0)
    assert _angle(M.EXTERNAL_ROTATION, u, R, None, yaw=FRO).status == MotionStatus.NOT_MEASURABLE
    straight_elbow = np.array([0.0, 0.0, -1.0])  # forearm continues the upper arm: elbow 0 deg
    assert _angle(M.EXTERNAL_ROTATION, u, R, straight_elbow).status == MotionStatus.NOT_MEASURABLE
    assert _angle(M.EXTERNAL_ROTATION, _u(45, 0), R, _forearm_rot(10)).angle_deg is None


# ---------------------------------------------------------------- camera view & quality
def test_view_classification_and_confidence_effect():
    assert [classify_view(y) for y in (None, 0.0, 45.0, 90.0)] == [
        CameraView.UNKNOWN,
        CameraView.FRONTAL,
        CameraView.OBLIQUE,
        CameraView.SAGITTAL,
    ]
    u = _u(80.0, 90.0)
    good = _angle(M.FLEXION, u, yaw=SAG)
    poor = _angle(M.FLEXION, u, yaw=FRO)  # forward motion is depth when viewed frontally
    unknown = _angle(M.FLEXION, u, yaw=None)
    assert good.confidence > unknown.confidence > poor.confidence
    assert good.status == MotionStatus.OK and poor.status != MotionStatus.OK
    assert poor.angle_deg is not None and any("poor" in r for r in poor.reasons)
    assert poor.angle_deg == pytest.approx(good.angle_deg)  # geometry same; only confidence differs

    a_good = _angle(M.ABDUCTION, _u(80.0, 0.0), yaw=FRO)
    a_poor = _angle(M.ABDUCTION, _u(80.0, 0.0), yaw=SAG)
    assert a_good.confidence > a_poor.confidence


def test_low_quality_withholds_the_angle():
    est = _angle(M.ABDUCTION, _u(60.0, 0.0), yaw=FRO, quality=0.2)
    assert est.status == MotionStatus.UNRELIABLE and est.angle_deg is None
    assert est.confidence < 0.25
    mid = _angle(M.ABDUCTION, _u(60.0, 0.0), yaw=FRO, quality=0.5)
    assert mid.status == MotionStatus.LOW_CONFIDENCE and mid.angle_deg is not None


def test_confidence_decreases_as_arm_leaves_defining_plane():
    on = _angle(M.FLEXION, _u(80.0, 90.0), yaw=SAG).confidence
    off = _angle(M.FLEXION, _u(80.0, 90.0 - 20.0), yaw=SAG).confidence
    assert on > off


def test_noisy_directions_stay_close_and_never_crash():
    rng = np.random.default_rng(3)
    errs = []
    for _ in range(200):
        u = _u(90.0, 0.0) + rng.normal(0.0, 0.03, 3)  # ~2 deg direction noise
        est = _angle(M.ABDUCTION, u / np.linalg.norm(u), yaw=FRO)
        if est.angle_deg is not None:
            errs.append(abs(est.angle_deg - 90.0))
    assert errs and max(errs) < 8.0 and float(np.mean(errs)) < 3.0


def test_conflicting_selection_vs_observed_pose():
    arm = _u(90.0, 0.0)  # patient abducts while FLEXION was selected
    assert _angle(M.FLEXION, arm, yaw=SAG).angle_deg is None
    s = suggest_movement(_seg(arm))
    assert s.movement == M.ABDUCTION


# ---------------------------------------------------------------- suggestion (separate)
@pytest.mark.parametrize(
    "elev,plane,expected",
    [(60.0, 90.0, M.FLEXION), (40.0, -90.0, M.EXTENSION), (60.0, 0.0, M.ABDUCTION),
     (30.0, 180.0, M.ADDUCTION), (120.0, 30.0, M.SCAPTION)],
)  # fmt: skip
def test_suggestion_unique_cases(elev, plane, expected):
    assert suggest_movement(_seg(_u(elev, plane))).movement == expected


def test_suggestion_is_ambiguous_for_forward_horizontal_arm():
    s = suggest_movement(_seg(_u(90.0, 90.0)))
    assert s.movement is None
    assert {M.FLEXION, M.HORIZONTAL_ADDUCTION} <= set(s.candidates)


def test_suggestion_neutral_and_never_rotation():
    s = suggest_movement(_seg(_u(0.0, 0.0), forearm=_forearm_rot(60.0)))
    assert s.movement is None and s.candidates == ()


# ---------------------------------------------------------------- view estimation
@pytest.mark.parametrize("yaw", [0.0, 30.0, 60.0, 90.0])
def test_subject_yaw_estimated_from_shoulder_line(yaw):
    body = make_neutral_standing_pose()
    a = np.radians(yaw)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    pts = [
        rot @ p for p in (body.left_shoulder, body.right_shoulder, body.left_hip, body.right_hip)
    ]
    frame = estimate_torso_frame(*pts)
    fixture_camera = CameraAxes(lateral=(1.0, 0.0, 0.0), depth=(0.0, 1.0, 0.0))
    assert estimate_subject_yaw_deg(frame, fixture_camera) == pytest.approx(yaw, abs=1e-6)


# ---------------------------------------------------------------- circumduction (compound)
def _circle(n: int, start: float = 0.0, span: float = 360.0, elev: float = 90.0):
    return [(elev, start + span * i / n) for i in range(n + 1)]


def test_circumduction_detected_on_full_circle_with_components():
    t = CircumductionTracker()
    st = None
    for e, p in _circle(72):
        wrapped = (p + 180.0) % 360.0 - 180.0
        st = t.update(e, wrapped)
    assert st is not None and st.detected and abs(st.sweep_deg) >= 359.0
    assert {M.ABDUCTION, M.FLEXION, M.EXTENSION} <= set(st.components)


def test_partial_arc_and_low_elevation_not_circumduction():
    t = CircumductionTracker()
    for e, p in _circle(30, span=180.0):
        st = t.update(e, p)
    assert not st.detected
    t.reset()
    for e, p in _circle(72, elev=10.0):
        st = t.update(e, (p + 180.0) % 360.0 - 180.0)
    assert not st.detected and st.sweep_deg == 0.0


def test_back_and_forth_between_planes_is_not_circumduction():
    t = CircumductionTracker()
    for _ in range(5):
        for p in list(range(0, 90, 10)) + list(range(90, 0, -10)):
            st = t.update(90.0, float(p))
    assert not st.detected and abs(st.sweep_deg) < 20.0


def test_tracking_loss_resets_circumduction():
    t = CircumductionTracker()
    for e, p in _circle(36, span=200.0):
        t.update(e, p)
    assert t.update(None, None).sweep_deg == 0.0


def test_unreliable_reports_real_cause_first_not_the_generic_note():
    est = _angle(M.ABDUCTION, _u(60.0, 0.0), yaw=FRO, quality=0.2)
    assert est.status == MotionStatus.UNRELIABLE
    assert est.reasons[0] == "below reporting floor"
    ok = _angle(M.ABDUCTION, _u(60.0, 0.0), yaw=SAG)
    assert ok.reasons[-1] == "confidence is a heuristic, not an accuracy bound"
    assert ok.reasons[0] != ok.reasons[-1]  # a specific reason (view) comes first
