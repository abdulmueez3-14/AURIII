"""Optional compound movement: circumduction.

Circumduction is NOT a single angle. It is detected as a sustained sweep of the plane of
elevation (the direction of the arm projected on the transverse plane) while the arm stays
elevated, and reported together with the movement components (sectors) that were traversed.
"""

from dataclasses import dataclass
from typing import Final

from auri_cv.movements.shoulder_movements import ShoulderMovement

MIN_ELEVATION_DEG: Final[float] = 30.0
SWEEP_REQUIRED_DEG: Final[float] = 300.0

# Plane-of-elevation sectors (0 = lateral, +90 = anterior), sign-adjusted for laterality
_SECTORS: Final[tuple[tuple[float, float, ShoulderMovement], ...]] = (
    (-45.0, 45.0, ShoulderMovement.ABDUCTION),
    (45.0, 135.0, ShoulderMovement.FLEXION),
    (-135.0, -45.0, ShoulderMovement.EXTENSION),
)


@dataclass(frozen=True)
class CircumductionState:
    sweep_deg: float  # signed, unwrapped plane-of-elevation travel since the arm was raised
    detected: bool
    components: tuple[ShoulderMovement, ...]  # movement components traversed


class CircumductionTracker:
    def __init__(
        self,
        min_elevation_deg: float = MIN_ELEVATION_DEG,
        sweep_required_deg: float = SWEEP_REQUIRED_DEG,
    ) -> None:
        self._min_elev = min_elevation_deg
        self._required = sweep_required_deg
        self.reset()

    def reset(self) -> None:
        self._prev: float | None = None
        self._sweep = 0.0
        self._visited: set[ShoulderMovement] = set()

    def update(self, elevation_deg: float | None, plane_deg: float | None) -> CircumductionState:
        """Feed elevation and plane of elevation; None/low elevation restarts the sweep."""
        if elevation_deg is None or plane_deg is None or elevation_deg < self._min_elev:
            self.reset()
            return CircumductionState(0.0, False, ())
        if self._prev is not None:
            delta = (plane_deg - self._prev + 180.0) % 360.0 - 180.0
            self._sweep += delta
        self._prev = plane_deg
        for lo, hi, mv in _SECTORS:
            if lo <= plane_deg < hi:
                self._visited.add(mv)
        order = tuple(m for _, _, m in _SECTORS if m in self._visited)
        return CircumductionState(self._sweep, abs(self._sweep) >= self._required, order)
