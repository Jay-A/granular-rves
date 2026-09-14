"""Tests for isotropic linear elasticity."""

import numpy as np
import ufl
from mpi4py import MPI

from dolfinx import fem
from dolfinx.mesh import create_unit_cube

from granular_rves.mechanics.constitutive.linear_elastic import (
    LinearElastic,
)


def test_linear_elastic_parameters() -> None:
    """Linear-elastic material parameters are stored correctly."""
    model = LinearElastic(
        youngs_modulus=1.0e6,
        poisson_ratio=0.3,
    )

    assert model.youngs_modulus == 1.0e6
    assert model.poisson_ratio == 0.3

    expected_shear_modulus = 1.0e6 / (2.0 * (1.0 + 0.3))
    expected_lame_first_parameter = (
        1.0e6
        * 0.3
        / ((1.0 + 0.3) * (1.0 - 2.0 * 0.3))
    )

    assert np.isclose(
        model.shear_modulus,
        expected_shear_modulus,
    )
    assert np.isclose(
        model.lame_first_parameter,
        expected_lame_first_parameter,
    )


def test_linear_elastic_rejects_nonpositive_youngs_modulus() -> None:
    """Linear elasticity rejects a nonpositive Young's modulus."""
    try:
        LinearElastic(
            youngs_modulus=0.0,
            poisson_ratio=0.3,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError.")


def test_linear_elastic_rejects_invalid_poisson_ratio() -> None:
    """Linear elasticity rejects inadmissible Poisson's ratios."""
    for poisson_ratio in (-1.0, 0.5, 1.0):
        try:
            LinearElastic(
                youngs_modulus=1.0e6,
                poisson_ratio=poisson_ratio,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError.")


def test_linear_elastic_stress_expression() -> None:
    """Linear elasticity returns a three-dimensional stress tensor."""
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

    strain = ufl.sym(ufl.grad(displacement))

    model = LinearElastic(
        youngs_modulus=1.0e6,
        poisson_ratio=0.3,
    )

    stress = model.stress(strain)

    assert stress.ufl_shape == (3, 3)
