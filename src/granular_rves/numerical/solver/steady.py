"""Steady-state finite-element solver."""

from __future__ import annotations

from typing import Any

from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc


class SteadySolver:
    """Solve a steady linear finite-element mechanics problem.

    The solver orchestrates the finite-element objects produced by the
    formulation and boundary-condition layers. It does not define the
    mechanical constitutive equations, kinematics, mesh, or boundary
    regions.

    The steady finite-element problem is represented as

    .. math::

        a(\\mathbf{u}, \\mathbf{v}) = L(\\mathbf{v}),

    where ``a`` is the bilinear form representing internal virtual work
    and ``L`` is the linear form representing external virtual work.

    Parameters
    ----------
    formulation
        Finite-element formulation providing the bilinear and linear forms
        and the associated function space and mesh.
    boundary_conditions
        DOLFINx Dirichlet boundary conditions to apply to the finite-element
        system.

    Notes
    -----
    The formulation is responsible for constructing the UFL variational
    forms. The numerical boundary layer is responsible for translating
    problem-level boundary definitions into DOLFINx boundary conditions.

    This class is responsible for assembling and solving the resulting
    finite-element system.

    The current implementation uses a PETSc direct LU solver. This is
    appropriate for the small systems targeted by the current MVP and can
    be replaced or extended with other PETSc solver configurations as
    scalability requirements emerge.
    """

    def __init__(
        self,
        formulation: Any,
        boundary_conditions: list[Any] | tuple[Any, ...],
    ) -> None:
        """Initialize the steady-state solver.

        Parameters
        ----------
        formulation
            Finite-element formulation providing the bilinear and linear
            forms, function space, and computational mesh.
        boundary_conditions
            DOLFINx Dirichlet boundary conditions to apply during matrix
            assembly and right-hand-side construction.
        """
        self.formulation = formulation
        self.boundary_conditions = list(boundary_conditions)

    def solve(self) -> fem.Function:
        """Solve the steady finite-element problem.

        Returns
        -------
        dolfinx.fem.Function
            The solved displacement field.

        Raises
        ------
        RuntimeError
            If the PETSc linear solver does not converge.
        """
        bilinear_form = fem.form(
            self.formulation.bilinear_form()
        )
        linear_form = fem.form(
            self.formulation.linear_form()
        )

        matrix = petsc.assemble_matrix(
            bilinear_form,
            bcs=self.boundary_conditions,
        )
        matrix.assemble()

        vector = petsc.assemble_vector(linear_form)

        petsc.apply_lifting(
            vector,
            [bilinear_form],
            [self.boundary_conditions],
        )
        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        petsc.set_bc(
            vector,
            self.boundary_conditions,
        )

        solution = fem.Function(
            self.formulation.function_space
        )

        solver = PETSc.KSP().create(
            self.formulation.mesh.comm
        )
        solver.setOperators(matrix)
        solver.setType(PETSc.KSP.Type.PREONLY)

        preconditioner = solver.getPC()
        preconditioner.setType(PETSc.PC.Type.LU)

        solver.solve(
            vector,
            solution.x.petsc_vec,
        )
        solution.x.scatter_forward()

        if solver.getConvergedReason() <= 0:
            raise RuntimeError(
                "Steady-state linear solver failed to converge."
            )

        return solution


