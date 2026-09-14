"""Tests for small-strain kinematics."""

import numpy as np
from mpi4py import MPI

from dolfinx import fem
from dolfinx.mesh import create_unit_cube

from granular_rves.mechanics.kinematics.small_strain import (
    SmallStrainKinematics,
)


def test_small_strain_accepts_empty_parameters() -> None:
    """Small-strain kinematics accepts an empty parameter mapping."""
    model = SmallStrainKinematics(parameters={})

    assert model.parameters == {}


def test_small_strain_rejects_parameters() -> None:
    """Small-strain kinematics rejects unsupported parameters."""
    try:
        SmallStrainKinematics(parameters={"example": 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError.")


def test_small_strain() -> None:
    """Small-strain kinematics returns a symmetric displacement gradient."""
    mesh = create_unit_cube(
        MPI.COMM_SELF,
        1,
        1,
        1,
    )

    element = ("Lagrange", 1, (mesh.geometry.dim,))
    function_space = fem.functionspace(mesh, element)

    displacement = fem.Function(function_space)

    displacement.interpolate(
        lambda x: np.vstack(
            (
                x[0],
                x[1],
                x[2],
            )
        )
    )

    model = SmallStrainKinematics()
    strain = model.strain(displacement)

    assert strain.ufl_shape == (3, 3)
