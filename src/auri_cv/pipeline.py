"""Per-frame shoulder analysis pipeline composing all engine stages.

PoseObservation -> quality gate -> torso frame -> segments -> joint-angle solver
-> EMA filter -> calibrated channel (baseline, deadband) -> repetition FSM,
plus compensation versus the baseline torso frame.

No video, GUI or file I/O here. Raw solver output is returned alongside, and never
replaced by, filtered/relative/display values.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from auri_cv.anatomy.segments import ArmLaterality, ArmSegmentState, estimate_arm_segments_from_pose
from auri_cv.calibration import CalibratedChannel, ChannelReading, ChannelStatus
from auri_cv.filtering import ExponentialFilter
from auri_cv.frames.torso_frame import (
    MEDIAPIPE_WORLD_UP,
    TorsoFrame,
    estimate_torso_frame_from_pose,
    estimate_torso_frame_shoulders_only,
)
from auri_cv.geometry.exceptions import GeometryError
from auri_cv.kinematics import RawKinematics, solve_arm_kinematics
from auri_cv.movements import (
    MEDIAPIPE_CAMERA,
    CameraAxes,
    CompensationMetrics,
    MotionEstimate,
    MotionStatus,
    MovementSuggestion,
    RepetitionFSM,
    RepState,
    ShoulderMovement,
    compute_compensation,
    estimate_motion,
    estimate_subject_yaw_deg,
    shoulder_height_above_hips,
    suggest_movement,
)
from auri_cv.pose.data_models import PoseObservation
from auri_cv.quality.landmark_quality import (
    LandmarkQualityGate,
    QualityAssessmentResult,
    QualityStatus,
)


class TorsoMode(StrEnum):
    """Reference for the superior axis. Fixed per session so baseline and live frames agree."""

    HIPS = "HIPS"  # measured spine direction (needs both hips visible); full compensation
    SHOULDERS_ONLY_ASSUMED_VERTICAL = "SHOULDERS_ONLY_ASSUMED_VERTICAL"  # see torso_frame


@dataclass(frozen=True)
class FrameResult:
    """Everything known about one frame. Fields are None when not observable."""

    timestamp_ms: float
    frame_index: int
    quality: QualityAssessmentResult
    raw: RawKinematics | None  # unsmoothed solver output
    filtered_elevation_deg: float | None  # EMA of the tracked metric (elevation, or the
    # selected movement's angle when `movement` is set); this is what is calibrated
    channel: ChannelReading  # relative excursion and display value for elevation
    rep_state: RepState
    rep_count: int
    compensation: CompensationMetrics | None  # vs. baseline; None until calibrated
    torso_mode: TorsoMode = TorsoMode.HIPS
    # Movement-specific layer (None when no movement was selected: legacy elevation tracking)
    motion: MotionEstimate | None = None
    suggestion: MovementSuggestion | None = None  # hint only; never overrides the selection
    # Every movement's estimate for this frame (only with compute_all_movements=True). Each is
    # judged against its own definition, so most are NOT_MEASURABLE at any one time.
    all_motions: dict[ShoulderMovement, MotionEstimate] | None = None


class ShoulderAnalysisPipeline:
    """Stateful single-arm pipeline tracking humerothoracic elevation.

    The first frames after construction/`recalibrate()` are the calibration phase: the
    subject must hold the clinician-approved starting posture. Elevation is the tracked
    channel; flexion/abduction are reported in `raw` only.
    """

    def __init__(
        self,
        laterality: ArmLaterality | str = ArmLaterality.RIGHT,
        quality_gate: LandmarkQualityGate | None = None,
        channel: CalibratedChannel | None = None,
        fsm: RepetitionFSM | None = None,
        smoothing: ExponentialFilter | None = None,
        torso_mode: TorsoMode = TorsoMode.HIPS,
        up_world: tuple[float, float, float] = MEDIAPIPE_WORLD_UP,
        movement: ShoulderMovement | None = None,
        camera_axes: CameraAxes = MEDIAPIPE_CAMERA,
        compute_all_movements: bool = False,
    ) -> None:
        self._all = compute_all_movements
        self._movement = movement
        self._camera = camera_axes
        self._lat = ArmLaterality(laterality)
        self._mode = torso_mode
        self._up = np.array(up_world, dtype=np.float64)
        self._gate = quality_gate or LandmarkQualityGate()
        self._channel = channel or CalibratedChannel()
        self._fsm = fsm or RepetitionFSM()
        self._filter = smoothing or ExponentialFilter()
        self._baseline_frame: TorsoFrame | None = None
        self._baseline_height_m: float | None = None

    @property
    def fsm(self) -> RepetitionFSM:
        return self._fsm

    @property
    def channel(self) -> CalibratedChannel:
        return self._channel

    def recalibrate(self) -> None:
        """Discards the baseline and restarts calibration. Completed reps are kept."""
        self._channel.recalibrate()
        self._filter.reset()
        self._fsm.reset()
        self._baseline_frame = None
        self._baseline_height_m = None

    def process(self, pose: PoseObservation) -> FrameResult:
        hips = self._mode == TorsoMode.HIPS
        quality = self._gate.assess(pose, target_arm=self._lat.value, require_hips=hips)

        frame: TorsoFrame | None = None
        segment: ArmSegmentState | None = None
        if quality.is_usable_for_kinematics:
            frame = self._torso_frame(pose)
            if frame is not None:
                segment = estimate_arm_segments_from_pose(pose, frame, self._lat)

        raw: RawKinematics | None = None
        if segment is not None:
            raw = solve_arm_kinematics(segment, pose.timestamp_ms)

        # Geometry failures (degenerate torso/segments) count as lost tracking.
        status = quality.status
        if raw is None and status != QualityStatus.RECALIBRATION_REQUIRED:
            status = QualityStatus.UNAVAILABLE

        motion: MotionEstimate | None = None
        suggestion: MovementSuggestion | None = None
        metric: float | None = raw.elevation_deg if raw is not None else None
        pose_gated = False
        if self._movement is not None and segment is not None and frame is not None:
            motion = estimate_motion(
                self._movement,
                segment,
                quality_score=quality.quality_score,
                camera_yaw_deg=estimate_subject_yaw_deg(frame, self._camera),
                timestamp_ms=pose.timestamp_ms,
            )
            suggestion = suggest_movement(segment)
            # The tracked metric is the SELECTED movement's own angle; withheld angles are None.
            has_angle = motion.status in (MotionStatus.OK, MotionStatus.LOW_CONFIDENCE)
            metric = motion.angle_deg if has_angle else None
            pose_gated = not has_angle
        elif self._movement is not None:
            metric = None

        all_motions: dict[ShoulderMovement, MotionEstimate] | None = None
        if self._all and segment is not None and frame is not None:
            yaw = estimate_subject_yaw_deg(frame, self._camera)
            all_motions = {
                m: estimate_motion(
                    m,
                    segment,
                    quality_score=quality.quality_score,
                    camera_yaw_deg=yaw,
                    timestamp_ms=pose.timestamp_ms,
                )
                for m in ShoulderMovement
            }

        filtered = self._filter.update(metric)
        reading = self._channel.update(filtered, status, pose_gated=pose_gated)

        if self._baseline_frame is None and self._channel.baseline is not None:
            self._capture_baseline(pose, frame)

        t_s = pose.timestamp_ms / 1000.0
        excursion = reading.relative_deg if reading.status == ChannelStatus.OK else None
        rep_state = self._fsm.update(excursion, t_s)

        return FrameResult(
            timestamp_ms=pose.timestamp_ms,
            frame_index=pose.frame_index,
            quality=quality,
            raw=raw,
            filtered_elevation_deg=filtered,
            channel=reading,
            rep_state=rep_state,
            rep_count=self._fsm.rep_count,
            compensation=self._compensation(pose, frame, reading),
            torso_mode=self._mode,
            motion=motion,
            suggestion=suggestion,
            all_motions=all_motions,
        )

    def _torso_frame(self, pose: PoseObservation) -> TorsoFrame | None:
        if self._mode == TorsoMode.HIPS:
            return estimate_torso_frame_from_pose(pose)
        s_l = pose.get_world_point("LEFT_SHOULDER")
        s_r = pose.get_world_point("RIGHT_SHOULDER")
        if s_l is None or s_r is None:
            return None
        try:
            return estimate_torso_frame_shoulders_only(s_l, s_r, self._up)
        except GeometryError:
            return None

    def _capture_baseline(self, pose: PoseObservation, frame: TorsoFrame | None) -> None:
        """Stores the torso frame of the frame that completed calibration (resting pose)."""
        shoulder = pose.get_world_point(f"{self._lat.name}_SHOULDER")
        if frame is None or shoulder is None:
            return
        self._baseline_frame = frame
        self._baseline_height_m = shoulder_height_above_hips(shoulder, frame)

    def _compensation(
        self, pose: PoseObservation, frame: TorsoFrame | None, reading: ChannelReading
    ) -> CompensationMetrics | None:
        if (
            self._mode != TorsoMode.HIPS  # lean/hiking need measured hips
            or reading.status != ChannelStatus.OK
            or frame is None
            or self._baseline_frame is None
            or self._baseline_height_m is None
        ):
            return None
        shoulder: np.ndarray | None = pose.get_world_point(f"{self._lat.name}_SHOULDER")
        if shoulder is None:
            return None
        return compute_compensation(frame, self._baseline_frame, shoulder, self._baseline_height_m)
