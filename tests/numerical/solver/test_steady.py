"""Tests for the steady-state finite-element solver."""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

from granular_rves.numerical.solver.steady import SteadySolver


class _TestFormulation:
    """Minimal scalar finite-element formulation for solver testing.

    This test formulation mirrors the interface expected by
    :class:`~granular_rves.numerical.solver.steady.SteadySolver`
    without depending on the production mechanics formulation.
    """

    def __init__(self) -> None:
        """Initialize the test finite-element problem."""
        self.mesh = mesh.create_unit_square(
            MPI.COMM_WORLD,
            2,
            2,
        )
        self.function_space = fem.functionspace(
            self.mesh,
            ("Lagrange", 1),
        )

    def bilinear_form(self):
        """Return a simple Poisson-type bilinear form.

        Returns
        -------
        ufl.Form
            Bilinear form representing the Laplacian operator.
        """
        trial = ufl.TrialFunction(self.function_space)
        test = ufl.TestFunction(self.function_space)

        return ufl.inner(
            ufl.grad(trial),
            ufl.grad(test),
        ) * ufl.dx(domain=self.mesh)

    def linear_form(self):
        """Return a zero external-load linear form."""
        test = ufl.TestFunction(self.function_space)
        zero = fem.Function(self.function_space)
        zero.x.array[:] = 0.0

        return zero * test * ufl.dx(domain=self.mesh)


def make_boundary_condition(formulation: _TestFormulation):
    """Create a homogeneous Dirichlet condition on the left boundary."""
    boundary_dofs = fem.locate_dofs_geometrical(
        formulation.function_space,
        lambda x: np.isclose(x[0], 0.0),
    )

    return fem.dirichletbc(
        PETSc.ScalarType(0.0),
        boundary_dofs,
        formulation.function_space,
    )


def test_steady_solver_returns_solution() -> None:
    """Verify that the steady solver returns a finite-element function."""
    formulation = _TestFormulation()
    boundary_condition = make_boundary_condition(formulation)

    solver = SteadySolver(
        formulation=formulation,
        boundary_conditions=[boundary_condition],
    )

    solution = solver.solve()

    assert isinstance(solution, fem.Function)
    assert solution.function_space == formulation.function_space


def test_steady_solver_preserves_zero_solution() -> None:
    """Verify that zero loading and zero boundary values give zero solution."""
    formulation = _TestFormulation()
    boundary_condition = make_boundary_condition(formulation)

    solver = SteadySolver(
        formulation=formulation,
        boundary_conditions=[boundary_condition],
    )

    solution = solver.solve()

    assert np.allclose(
        solution.x.array,
        0.0,
    )


def test_steady_solver_with_constraint_returns_solution() -> None:
    """Verify that a scalar constraint can be included in the steady solve."""
    formulation = _TestFormulation()
    boundary_condition = make_boundary_condition(formulation)

    trial = ufl.TrialFunction(formulation.function_space)
    constraint = trial * ufl.dx(domain=formulation.mesh)

    solver = SteadySolver(
        formulation=formulation,
        boundary_conditions=[boundary_condition],
        constraints=[constraint],
    )

    solution = solver.solve()

    assert isinstance(solution, fem.Function)
    assert solution.function_space == formulation.function_space
    assert np.allclose(
        solution.x.array,
        0.0,
    )


def test_steady_solver_responds_to_nonzero_loading() -> None:
    """Verify that a nonzero load produces a nonzero solution."""
    formulation = _TestFormulation()
    boundary_condition = make_boundary_condition(formulation)

    def linear_form():
        test = ufl.TestFunction(formulation.function_space)
        return 1.0 * test * ufl.dx(domain=formulation.mesh)

    formulation.linear_form = linear_form

    solver = SteadySolver(
        formulation=formulation,
        boundary_conditions=[boundary_condition],
    )

    solution = solver.solve()

    assert isinstance(solution, fem.Function)
    assert np.any(np.abs(solution.x.array) > 1.0e-12)


