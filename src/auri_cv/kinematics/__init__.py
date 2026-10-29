"""Joint-angle kinematics solvers for AURI."""

from auri_cv.kinematics.elevation import calculate_humerothoracic_elevation
from auri_cv.kinematics.projections import (
    calculate_coronal_abduction,
    calculate_plane_of_elevation,
    calculate_sagittal_flexion,
)
from auri_cv.kinematics.solver import RawKinematics, solve_arm_kinematics

__all__ = [
    "RawKinematics",
    "calculate_coronal_abduction",
    "calculate_humerothoracic_elevation",
    "calculate_plane_of_elevation",
    "calculate_sagittal_flexion",
    "solve_arm_kinematics",
]
