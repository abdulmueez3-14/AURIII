"""Data models for 2D and 3D pose landmarks and observations."""

from enum import IntEnum
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class LandmarkIndex(IntEnum):
    """Standard MediaPipe 33-landmark skeletal indices."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Key anatomical landmark subsets
TORSO_LANDMARK_NAMES: Final[tuple[str, ...]] = (
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
)

RIGHT_ARM_LANDMARK_NAMES: Final[tuple[str, ...]] = (
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
)

LEFT_ARM_LANDMARK_NAMES: Final[tuple[str, ...]] = (
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
)


class Landmark2D(BaseModel):
    """Normalized 2D image landmark observed on screen."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Anatomical landmark name, e.g. RIGHT_SHOULDER")
    index: int = Field(ge=0, le=32, description="MediaPipe landmark index")
    x: float = Field(description="Normalized image x-coordinate [0.0, 1.0]")
    y: float = Field(description="Normalized image y-coordinate [0.0, 1.0]")
    z: float = Field(description="Relative landmark depth")
    visibility: float = Field(ge=0.0, le=1.0, default=1.0, description="Estimated visibility score")
    presence: float = Field(ge=0.0, le=1.0, default=1.0, description="Estimated presence score")

    def to_pixel_coords(self, width: int, height: int) -> tuple[int, int]:
        """Maps normalized (x, y) coordinates to integer pixel coordinates."""
        return int(round(self.x * width)), int(round(self.y * height))


class Landmark3D(BaseModel):
    """Metric 3D world landmark expressed in approximate meters."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Anatomical landmark name, e.g. RIGHT_SHOULDER")
    index: int = Field(ge=0, le=32, description="MediaPipe landmark index")
    x: float = Field(description="World x coordinate in meters")
    y: float = Field(description="World y coordinate in meters")
    z: float = Field(description="World z coordinate in meters")
    visibility: float = Field(ge=0.0, le=1.0, default=1.0, description="Estimated visibility score")
    presence: float = Field(ge=0.0, le=1.0, default=1.0, description="Estimated presence score")

    def to_numpy(self) -> np.ndarray:
        """Returns the 3D position as a 1D float64 numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)


class PoseObservation(BaseModel):
    """Complete observation packet for a single video frame."""

    model_config = ConfigDict(frozen=True)

    timestamp_ms: float = Field(ge=0.0, description="Monotonic frame timestamp in milliseconds")
    frame_index: int = Field(ge=0, default=0, description="Sequential video frame index")
    is_detected: bool = Field(description="Whether a human pose was successfully detected")
    landmarks_2d: dict[str, Landmark2D] = Field(
        default_factory=dict, description="2D normalized landmarks by anatomical name"
    )
    landmarks_3d: dict[str, Landmark3D] = Field(
        default_factory=dict, description="3D world landmarks by anatomical name"
    )

    def get_landmark_2d(self, name: str) -> Landmark2D | None:
        """Retrieves 2D landmark by name, or None if missing."""
        return self.landmarks_2d.get(name)

    def get_landmark_3d(self, name: str) -> Landmark3D | None:
        """Retrieves 3D landmark by name, or None if missing."""
        return self.landmarks_3d.get(name)

    def get_world_point(self, name: str) -> np.ndarray | None:
        """Retrieves the 3D world coordinates as a numpy array, or None if missing."""
        lm = self.get_landmark_3d(name)
        if lm is None:
            return None
        return lm.to_numpy()

    @classmethod
    def create_empty(cls, timestamp_ms: float = 0.0, frame_index: int = 0) -> "PoseObservation":
        """Creates an empty observation for undetected or missing frames."""
        return cls(
            timestamp_ms=timestamp_ms,
            frame_index=frame_index,
            is_detected=False,
            landmarks_2d={},
            landmarks_3d={},
        )
