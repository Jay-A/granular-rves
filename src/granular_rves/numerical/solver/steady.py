"""Steady-state finite-element solver.

This module orchestrates assembly and solution of the steady finite-element
system, including optional global rigid-body constraints enforced with
Lagrange multipliers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc


class SteadySolver:
    """Solve a steady finite-element mechanics problem.

    The solver orchestrates the finite-element objects produced by the
    formulation and boundary-condition layers. It does not define the
    mechanical constitutive equations, kinematics, mesh, or boundary
    regions.

    An unconstrained problem is represented by

    .. math::

        K \\mathbf{u} = \\mathbf{f}.

    When rigid-body constraint functionals are supplied, the solver
    constructs the augmented system

    .. math::

        \\begin{bmatrix}
        K & C^T \\\\
        C & 0
        \\end{bmatrix}
        \\begin{bmatrix}
        \\mathbf{u} \\\\
        \\boldsymbol{\\lambda}
        \\end{bmatrix}
        =
        \\begin{bmatrix}
        \\mathbf{f} \\\\
        0
        \\end{bmatrix}.

    Parameters
    ----------
    formulation
        Finite-element formulation providing the bilinear and linear
        forms and the associated function space and mesh.
    boundary_conditions
        DOLFINx Dirichlet boundary conditions to apply to the displacement
        system.
    constraints
        Optional sequence of UFL linear constraint functionals acting on
        the displacement field. Each functional contributes one Lagrange
        multiplier to the augmented system.

    Notes
    -----
    The formulation is responsible for constructing the UFL variational
    forms. The numerical boundary layer is responsible for translating
    problem-level boundary definitions into DOLFINx boundary conditions.

    This class is responsible for assembling and solving the resulting
    finite-element system.

    The current implementation uses PETSc's direct LU solver. This is
    appropriate for the small systems targeted by the current MVP and can
    be replaced or extended with other PETSc solver configurations as
    scalability requirements emerge.
    """

    def __init__(
        self,
        formulation: Any,
        boundary_conditions: list[Any] | tuple[Any, ...],
        constraints: list[Any] | tuple[Any, ...] | None = None,
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
        constraints
            Optional UFL constraint functionals acting on the displacement
            field.
        """
        self.formulation = formulation
        self.boundary_conditions = list(boundary_conditions)
        self.constraints = [] if constraints is None else list(constraints)

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

        vector = self._assemble_rhs(
            linear_form,
            bilinear_form,
        )

        if not self.constraints:
            return self._solve_unconstrained(matrix, vector)

        constraint_rows = self._assemble_constraint_rows()

        augmented_matrix = self._create_augmented_matrix(
            matrix,
            constraint_rows,
        )

        augmented_rhs = self._create_augmented_rhs(
            augmented_matrix,
            vector,
        )

        solution = self._create_augmented_solution(
            augmented_matrix,
        )

        self._solve_augmented(
            augmented_matrix,
            augmented_rhs,
            solution,
        )

        displacement = fem.Function(
            self.formulation.function_space
        )

        self._extract_displacement(
            solution,
            displacement,
        )

        displacement.x.scatter_forward()

        return displacement

    def _assemble_rhs(
        self,
        linear_form: Any,
        bilinear_form: Any,
    ) -> PETSc.Vec:
        """Assemble the displacement right-hand side.

        A mathematically zero linear form can be simplified by UFL to a form
        that no longer carries function-space metadata. In that case
        DOLFINx cannot assemble it directly. The solver therefore creates
        the zero vector from the formulation's function space.

        Parameters
        ----------
        linear_form
            Compiled DOLFINx linear form.
        bilinear_form
            Compiled DOLFINx bilinear form used when applying Dirichlet
            lifting.

        Returns
        -------
        petsc4py.PETSc.Vec
            Assembled displacement-space right-hand-side vector.
        """
        if not linear_form.function_spaces:
            vector = self._create_zero_displacement_vector()
        else:
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

        return vector

    def _create_zero_displacement_vector(self) -> PETSc.Vec:
        """Create a zero vector associated with the displacement space.

        Returns
        -------
        petsc4py.PETSc.Vec
            Zero vector with the DOLFINx displacement-space layout,
            including the ghost entries required by DOLFINx.
        """
        vector = petsc.create_vector(
            self.formulation.function_space,
        )
        vector.set(0.0)
        return vector

    def _solve_unconstrained(
        self,
        matrix: PETSc.Mat,
        vector: PETSc.Vec,
    ) -> fem.Function:
        """Solve the ordinary unconstrained displacement system.

        Parameters
        ----------
        matrix
            Assembled stiffness matrix.
        vector
            Assembled right-hand-side vector.

        Returns
        -------
        dolfinx.fem.Function
            Solved displacement field.

        Raises
        ------
        RuntimeError
            If the PETSc solver does not converge.
        """
        solution = fem.Function(
            self.formulation.function_space
        )

        solver = PETSc.KSP().create(
            self.formulation.mesh.comm,
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

    def _assemble_constraint_rows(self) -> list[PETSc.Vec]:
        """Assemble the displacement-space constraint rows.

        Returns
        -------
        list of petsc4py.PETSc.Vec
            One assembled displacement-space vector for each constraint
            functional.
        """
        rows: list[PETSc.Vec] = []

        for constraint in self.constraints:
            form = fem.form(constraint)
            row = petsc.assemble_vector(form)

            row.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES,
                mode=PETSc.ScatterMode.REVERSE,
            )

            rows.append(row)

        return rows

    def _create_augmented_matrix(
        self,
        matrix: PETSc.Mat,
        constraint_rows: list[PETSc.Vec],
    ) -> PETSc.Mat:
        """Create the augmented PETSc saddle-point matrix.

        The augmented system has the block structure

        .. math::

            \\begin{bmatrix}
            K & C^T \\\\
            C & 0
            \\end{bmatrix},

        where ``K`` is the displacement stiffness matrix and ``C`` contains
        the assembled constraint functionals.

        Parameters
        ----------
        matrix
            Displacement stiffness matrix.
        constraint_rows
            Assembled displacement-space constraint vectors.

        Returns
        -------
        petsc4py.PETSc.Mat
            AIJ-format augmented saddle-point matrix.
        """
        comm = self.formulation.mesh.comm

        displacement_size = matrix.getSize()[0]
        number_of_constraints = len(constraint_rows)
        augmented_size = displacement_size + number_of_constraints

        augmented = PETSc.Mat().createAIJ(
            [
                augmented_size,
                augmented_size,
            ],
            comm=comm,
        )
        augmented.setUp()

        # Copy the displacement stiffness matrix into the upper-left block.
        row_start, row_end = matrix.getOwnershipRange()

        for row_index in range(row_start, row_end):
            columns, values = matrix.getRow(row_index)

            if len(columns) > 0:
                augmented.setValues(
                    row_index,
                    columns,
                    values,
                )

        # Insert the constraint matrix C and its transpose C^T directly.
        for constraint_index, row in enumerate(constraint_rows):
            constraint_row = displacement_size + constraint_index

            start, end = row.getOwnershipRange()

            for column in range(start, end):
                value = row.getValue(column)

                if value == 0.0:
                    continue

                augmented.setValue(
                    constraint_row,
                    column,
                    value,
                )
                augmented.setValue(
                    column,
                    constraint_row,
                    value,
                )

            # Keep an explicit structural zero on the multiplier diagonal.
            augmented.setValue(
                constraint_row,
                constraint_row,
                0.0,
            )

        augmented.assemble()

        return augmented

    def _create_augmented_rhs(
        self,
        matrix: PETSc.Mat,
        vector: PETSc.Vec,
    ) -> PETSc.Vec:
        """Create a vector compatible with the augmented matrix.

        Parameters
        ----------
        matrix
            Augmented saddle-point matrix.
        vector
            Displacement right-hand-side vector.

        Returns
        -------
        petsc4py.PETSc.Vec
            Flat augmented right-hand-side vector containing the
            displacement RHS followed by zero constraint entries.
        """
        rhs, _ = matrix.createVecs()
        rhs.set(0.0)

        start, end = vector.getOwnershipRange()
        values = vector.getValues(
            list(range(start, end)),
        )

        if end > start:
            rhs.setValues(
                list(range(start, end)),
                values,
            )

        rhs.assemble()

        return rhs

    def _create_augmented_solution(
        self,
        matrix: PETSc.Mat,
    ) -> PETSc.Vec:
        """Create a solution vector compatible with the augmented matrix.

        Parameters
        ----------
        matrix
            Augmented saddle-point matrix.

        Returns
        -------
        petsc4py.PETSc.Vec
            Flat augmented solution vector.
        """
        solution, _ = matrix.createVecs()
        solution.set(0.0)

        return solution

    def _extract_displacement(
        self,
        solution: PETSc.Vec,
        displacement: fem.Function,
    ) -> None:
        """Copy the displacement block from an augmented solution.

        Parameters
        ----------
        solution
            Flat augmented solution vector.
        displacement
            Function receiving the displacement degrees of freedom.
        """
        displacement_vector = displacement.x.petsc_vec

        start, end = displacement_vector.getOwnershipRange()

        if end <= start:
            return

        values = solution.getValues(
            list(range(start, end)),
        )

        displacement_vector.setValues(
            list(range(start, end)),
            np.asarray(values),
        )
        displacement_vector.assemble()

    def _solve_augmented(
        self,
        matrix: PETSc.Mat,
        rhs: PETSc.Vec,
        solution: PETSc.Vec,
    ) -> None:
        """Solve the augmented saddle-point system.

        Parameters
        ----------
        matrix
            AIJ-format augmented saddle-point matrix.
        rhs
            Flat augmented right-hand-side vector.
        solution
            Flat augmented solution vector.

        Raises
        ------
        RuntimeError
            If the PETSc solver does not converge.
        """
        matrix.assemble()

        solver = PETSc.KSP().create(
            self.formulation.mesh.comm,
        )
        solver.setOperators(matrix)
        solver.setType(PETSc.KSP.Type.PREONLY)

        preconditioner = solver.getPC()
        preconditioner.setType(PETSc.PC.Type.LU)

        solver.solve(
            rhs,
            solution,
        )

        reason = solver.getConvergedReason()

        if reason <= 0:
            raise RuntimeError(
                "Steady-state constrained linear solver failed "
                f"to converge (reason={reason}, "
                f"residual={solver.getResidualNorm():.6e})."
            )


