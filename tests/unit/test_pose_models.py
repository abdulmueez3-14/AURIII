"""Unit tests for pose observation data models."""

import numpy as np
import pytest
from pydantic import ValidationError

from auri_cv.pose.data_models import (
    Landmark2D,
    Landmark3D,
    LandmarkIndex,
    PoseObservation,
)


class TestLandmark2D:
    def test_landmark2d_instantiation(self):
        lm = Landmark2D(
            name="RIGHT_SHOULDER",
            index=12,
            x=0.45,
            y=0.30,
            z=-0.10,
            visibility=0.95,
            presence=0.98,
        )
        assert lm.name == "RIGHT_SHOULDER"
        assert lm.index == 12
        assert lm.x == 0.45
        assert lm.y == 0.30
        assert lm.visibility == 0.95

    def test_landmark2d_pixel_coords_mapping(self):
        lm = Landmark2D(
            name="NOSE",
            index=0,
            x=0.5,
            y=0.25,
            z=0.0,
            visibility=1.0,
            presence=1.0,
        )
        px, py = lm.to_pixel_coords(width=1920, height=1080)
        assert px == 960
        assert py == 270

    def test_landmark2d_immutability(self):
        lm = Landmark2D(
            name="NOSE",
            index=0,
            x=0.5,
            y=0.5,
            z=0.0,
            visibility=1.0,
            presence=1.0,
        )
        with pytest.raises(ValidationError):
            lm.x = 0.8  # Frozen model cannot be modified


class TestLandmark3D:
    def test_landmark3d_instantiation(self):
        lm = Landmark3D(
            name="LEFT_SHOULDER",
            index=11,
            x=0.20,
            y=-0.15,
            z=0.60,
            visibility=0.90,
            presence=0.92,
        )
        assert lm.name == "LEFT_SHOULDER"
        np.testing.assert_allclose(lm.to_numpy(), [0.20, -0.15, 0.60])

    def test_landmark3d_immutability(self):
        lm = Landmark3D(
            name="LEFT_ELBOW",
            index=13,
            x=0.30,
            y=0.0,
            z=0.30,
            visibility=0.85,
            presence=0.85,
        )
        with pytest.raises(ValidationError):
            lm.y = 1.0


class TestPoseObservation:
    def test_empty_observation(self):
        empty = PoseObservation.create_empty(timestamp_ms=100.0, frame_index=5)
        assert not empty.is_detected
        assert empty.timestamp_ms == 100.0
        assert empty.frame_index == 5
        assert len(empty.landmarks_2d) == 0
        assert len(empty.landmarks_3d) == 0
        assert empty.get_landmark_3d("RIGHT_SHOULDER") is None
        assert empty.get_world_point("RIGHT_SHOULDER") is None

    def test_valid_observation_lookup(self):
        lm2d = Landmark2D(
            name="RIGHT_SHOULDER",
            index=LandmarkIndex.RIGHT_SHOULDER,
            x=0.4,
            y=0.3,
            z=0.0,
            visibility=0.99,
            presence=0.99,
        )
        lm3d = Landmark3D(
            name="RIGHT_SHOULDER",
            index=LandmarkIndex.RIGHT_SHOULDER,
            x=-0.2,
            y=0.0,
            z=1.4,
            visibility=0.99,
            presence=0.99,
        )
        obs = PoseObservation(
            timestamp_ms=500.0,
            frame_index=15,
            is_detected=True,
            landmarks_2d={"RIGHT_SHOULDER": lm2d},
            landmarks_3d={"RIGHT_SHOULDER": lm3d},
        )
        assert obs.is_detected
        assert obs.get_landmark_2d("RIGHT_SHOULDER") == lm2d
        assert obs.get_landmark_3d("RIGHT_SHOULDER") == lm3d
        np.testing.assert_allclose(obs.get_world_point("RIGHT_SHOULDER"), [-0.2, 0.0, 1.4])
