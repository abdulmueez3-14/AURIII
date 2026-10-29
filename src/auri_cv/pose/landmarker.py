"""MediaPipe Tasks PoseLandmarker wrapper for AURI."""

import logging
import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
)

from auri_cv.pose.data_models import Landmark2D, Landmark3D, LandmarkIndex, PoseObservation

logger = logging.getLogger(__name__)

OFFICIAL_MODEL_URL: str = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
DEFAULT_MODEL_PATH: str = "models/pose_landmarker_lite.task"


def ensure_model_asset(model_path: str = DEFAULT_MODEL_PATH) -> Path:
    """Verifies that the model asset exists, or downloads it if absent.

    Args:
        model_path: Local path to the .task model file.

    Returns:
        Resolved Path to the verified model asset.

    Raises:
        FileNotFoundError: If model cannot be found or downloaded.
    """
    path = Path(model_path)
    if path.is_file() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading MediaPipe Pose Landmarker model from {OFFICIAL_MODEL_URL}...")
    try:
        urllib.request.urlretrieve(OFFICIAL_MODEL_URL, str(path))
        logger.info(f"Successfully downloaded model to {path} ({path.stat().st_size} bytes)")
    except Exception as e:
        raise FileNotFoundError(
            f"MediaPipe model asset not found at {path} and failed to download from {OFFICIAL_MODEL_URL}: {e}"
        ) from e

    return path


def _score(value: float | None) -> float:
    """Confidence score; a missing value means unreported (1.0), but 0.0 is a real 0.0."""
    return 1.0 if value is None else float(value)


class AuriPoseLandmarker:
    """Production wrapper around Google MediaPipe Tasks Vision PoseLandmarker."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        running_mode: VisionTaskRunningMode = VisionTaskRunningMode.IMAGE,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initializes PoseLandmarker with specified options.

        Args:
            model_path: Path to pose_landmarker_*.task model.
            running_mode: IMAGE, VIDEO, or LIVE_STREAM.
            min_detection_confidence: Detection confidence threshold.
            min_presence_confidence: Presence confidence threshold.
            min_tracking_confidence: Tracking confidence threshold.
        """
        resolved_path = ensure_model_asset(model_path)
        self.model_path = str(resolved_path)
        self.running_mode = running_mode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=self.running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )

        self._landmarker: PoseLandmarker | None = PoseLandmarker.create_from_options(options)

    def detect_image(
        self,
        rgb_image: np.ndarray,
        timestamp_ms: float = 0.0,
        frame_index: int = 0,
        is_mirrored: bool = False,
    ) -> PoseObservation:
        """Processes a single RGB image in IMAGE running mode.

        Args:
            rgb_image: Numpy array of shape (H, W, 3) in RGB format (uint8).
            timestamp_ms: Monotonic timestamp in milliseconds.
            frame_index: Sequential frame index.
            is_mirrored: If True, indicates video input was horizontally mirrored.

        Returns:
            Structured, immutable PoseObservation.
        """
        if self._landmarker is None:
            raise RuntimeError("AuriPoseLandmarker has been closed.")

        if self.running_mode != VisionTaskRunningMode.IMAGE:
            raise ValueError(
                f"detect_image requires IMAGE running mode, current is {self.running_mode}"
            )

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result: PoseLandmarkerResult = self._landmarker.detect(mp_image)
        return self._parse_result(
            result, timestamp_ms=timestamp_ms, frame_index=frame_index, is_mirrored=is_mirrored
        )

    def detect_video_frame(
        self,
        rgb_image: np.ndarray,
        timestamp_ms: float,
        frame_index: int = 0,
        is_mirrored: bool = False,
    ) -> PoseObservation:
        """Processes a video frame with monotonically increasing timestamps.

        Args:
            rgb_image: Numpy array of shape (H, W, 3) in RGB format (uint8).
            timestamp_ms: Monotonically increasing frame timestamp in milliseconds.
            frame_index: Sequential frame index.
            is_mirrored: If True, indicates video input was horizontally mirrored.

        Returns:
            Structured, immutable PoseObservation.
        """
        if self._landmarker is None:
            raise RuntimeError("AuriPoseLandmarker has been closed.")

        if self.running_mode != VisionTaskRunningMode.VIDEO:
            raise ValueError(
                f"detect_video_frame requires VIDEO running mode, current is {self.running_mode}"
            )

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # MediaPipe video detection requires integer milliseconds
        ts_int = int(round(timestamp_ms))
        result: PoseLandmarkerResult = self._landmarker.detect_for_video(mp_image, ts_int)
        return self._parse_result(
            result, timestamp_ms=timestamp_ms, frame_index=frame_index, is_mirrored=is_mirrored
        )

    def _parse_result(
        self,
        result: PoseLandmarkerResult,
        timestamp_ms: float,
        frame_index: int,
        is_mirrored: bool,
    ) -> PoseObservation:
        """Converts raw MediaPipe Task result into our typed PoseObservation."""
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return PoseObservation.create_empty(timestamp_ms=timestamp_ms, frame_index=frame_index)

        raw_2d = result.pose_landmarks[0]
        raw_3d = (
            result.pose_world_landmarks[0]
            if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0
            else None
        )

        landmarks_2d: dict[str, Landmark2D] = {}
        landmarks_3d: dict[str, Landmark3D] = {}

        for i, lm in enumerate(raw_2d):
            try:
                name = LandmarkIndex(i).name
            except ValueError:
                continue

            x_2d = float(1.0 - lm.x) if is_mirrored else float(lm.x)
            vis = _score(getattr(lm, "visibility", None))
            pres = _score(getattr(lm, "presence", None))

            landmarks_2d[name] = Landmark2D(
                name=name,
                index=i,
                x=x_2d,
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(np.clip(vis, 0.0, 1.0)),
                presence=float(np.clip(pres, 0.0, 1.0)),
            )

        if raw_3d is not None:
            for i, lm in enumerate(raw_3d):
                try:
                    name = LandmarkIndex(i).name
                except ValueError:
                    continue

                vis = _score(getattr(lm, "visibility", None))
                pres = _score(getattr(lm, "presence", None))

                landmarks_3d[name] = Landmark3D(
                    name=name,
                    index=i,
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=float(np.clip(vis, 0.0, 1.0)),
                    presence=float(np.clip(pres, 0.0, 1.0)),
                )

        return PoseObservation(
            timestamp_ms=timestamp_ms,
            frame_index=frame_index,
            is_detected=True,
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
        )

    def close(self) -> None:
        """Closes the underlying MediaPipe landmarker and releases C++ resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "AuriPoseLandmarker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
