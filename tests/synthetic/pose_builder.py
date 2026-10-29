"""Builds PoseObservation sequences from analytic geometry (fixture frame: Z up, subject
right = world -X, anterior = world -Y)."""

import numpy as np

from auri_cv.pose.data_models import Landmark3D, LandmarkIndex, PoseObservation
from tests.synthetic.fixtures import make_neutral_standing_pose

UPPER_ARM_M = 0.30
FOREARM_M = 0.25


def make_observation(
    elevation_deg: float,
    t_ms: float,
    frame_index: int,
    *,
    plane_deg: float = 0.0,  # 0 = abduction (coronal), 90 = forward flexion (sagittal)
    right_shoulder_lift_m: float = 0.0,
    trunk_roll_deg: float = 0.0,  # rotation about world Y: lateral lean toward subject right
    noise_m: float = 0.0,
    rng: np.random.Generator | None = None,
    drop: bool = False,
    visibility: float = 0.95,
    drop_hips: bool = False,
    arm: str = "right",
    forearm_dir_torso: tuple[float, float, float] | None = None,
    body_yaw_deg: float = 0.0,
) -> PoseObservation:
    """`arm` raised `elevation_deg` from hanging, in the given plane (torso-relative).

    Torso axes in the fixture world: x_T = -X (subject right), y_T = -Y (anterior), z_T = +Z.
    `forearm_dir_torso` (torso coords) overrides the straight forearm (elbow bent).
    `body_yaw_deg` turns the whole body about the vertical (camera view changes).
    """
    if drop:
        return PoseObservation.create_empty(t_ms, frame_index)

    body = make_neutral_standing_pose()
    s_r = body.right_shoulder + np.array([0.0, 0.0, right_shoulder_lift_m])
    s_l = body.left_shoulder
    e, p = np.radians(elevation_deg), np.radians(plane_deg)
    lateral = 1.0 if arm == "right" else -1.0  # lateral direction in torso x (right arm: +x_T)
    d_t = np.array([lateral * np.sin(e) * np.cos(p), np.sin(e) * np.sin(p), -np.cos(e)])
    to_world = np.diag([-1.0, -1.0, 1.0])  # torso -> fixture world
    d = to_world @ d_t
    f = d if forearm_dir_torso is None else to_world @ np.asarray(forearm_dir_torso, dtype=float)
    shoulder = s_r if arm == "right" else s_l
    elbow = shoulder + UPPER_ARM_M * d
    wrist = elbow + FOREARM_M * f

    pts = {
        "LEFT_SHOULDER": body.left_shoulder,
        "RIGHT_SHOULDER": s_r,
        "LEFT_HIP": body.left_hip,
        "RIGHT_HIP": body.right_hip,
        "LEFT_ELBOW": body.left_elbow,
        "LEFT_WRIST": body.left_wrist,
        "RIGHT_ELBOW": body.right_elbow,
        "RIGHT_WRIST": body.right_wrist,
    }
    side = arm.upper()
    pts[f"{side}_ELBOW"] = elbow
    pts[f"{side}_WRIST"] = wrist
    if trunk_roll_deg:
        a = np.radians(-trunk_roll_deg)
        c, s = np.cos(a), np.sin(a)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pivot = 0.5 * (body.left_hip + body.right_hip)
        pts = {k: pivot + rot @ (v - pivot) for k, v in pts.items()}

    if body_yaw_deg:
        a = np.radians(body_yaw_deg)
        c, s_ = np.cos(a), np.sin(a)
        yaw = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        pivot = 0.5 * (body.left_hip + body.right_hip)
        pts = {k: pivot + yaw @ (v - pivot) for k, v in pts.items()}

    if drop_hips:
        pts = {k: v for k, v in pts.items() if "HIP" not in k}
    lms: dict[str, Landmark3D] = {}
    for name, v in pts.items():
        if noise_m and rng is not None:
            v = v + rng.normal(0.0, noise_m, 3)
        lms[name] = Landmark3D(
            name=name,
            index=int(LandmarkIndex[name]),
            x=float(v[0]),
            y=float(v[1]),
            z=float(v[2]),
            visibility=visibility,
            presence=visibility,
        )
    return PoseObservation(
        timestamp_ms=t_ms, frame_index=frame_index, is_detected=True, landmarks_3d=lms
    )


def elevation_profile(
    peak_deg: float, rate_dps: float = 45.0, hold_frames: int = 10, fps: float = 30.0
) -> list[float]:
    """Triangle 0 -> peak -> 0 with a hold at the top (no resting padding)."""
    step = rate_dps / fps
    up = [min(peak_deg, i * step) for i in range(int(peak_deg / step) + 2)]
    return up + [peak_deg] * hold_frames + up[::-1]
