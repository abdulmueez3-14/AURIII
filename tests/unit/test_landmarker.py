"""Unit tests for AuriPoseLandmarker wrapper."""

import numpy as np
import pytest
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from auri_cv.pose.landmarker import (
    AuriPoseLandmarker,
    ensure_model_asset,
)


class TestPoseLandmarkerInitialization:
    def test_model_asset_exists(self):
        model_path = ensure_model_asset()
        assert model_path.is_file()
        assert model_path.stat().st_size > 0

    def test_landmarker_context_manager(self):
        with AuriPoseLandmarker() as landmarker:
            assert landmarker.running_mode == VisionTaskRunningMode.IMAGE
            blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
            obs = landmarker.detect_image(blank_image, timestamp_ms=0.0)
            assert not obs.is_detected
            assert len(obs.landmarks_2d) == 0

        # After exiting context, landmarker is closed
        with pytest.raises(RuntimeError, match="has been closed"):
            landmarker.detect_image(blank_image)

    def test_invalid_mode_mismatch_raises(self):
        with AuriPoseLandmarker(running_mode=VisionTaskRunningMode.IMAGE) as landmarker:
            blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
            with pytest.raises(ValueError, match="detect_video_frame requires VIDEO"):
                landmarker.detect_video_frame(blank_image, timestamp_ms=10.0)


def test_zero_confidence_is_preserved_not_promoted_to_one():
    from types import SimpleNamespace

    from auri_cv.pose.landmarker import AuriPoseLandmarker, _score

    assert _score(0.0) == 0.0
    assert _score(None) == 1.0
    assert _score(0.37) == 0.37

    lm = SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.0, presence=0.0)
    result = SimpleNamespace(pose_landmarks=[[lm]], pose_world_landmarks=[[lm]])
    landmarker = object.__new__(AuriPoseLandmarker)
    obs = landmarker._parse_result(result, timestamp_ms=0.0, frame_index=0, is_mirrored=False)
    nose = obs.get_landmark_3d("NOSE")
    assert nose is not None and nose.visibility == 0.0 and nose.presence == 0.0
