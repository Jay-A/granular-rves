"""Reaction-force response evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

from granular_rves.numerical.boundary.dirichlet import locate_dirichlet_dofs


def assemble_physical_residual(
    formulation: Any,
    displacement: fem.Function,
    constraint_rows: list[PETSc.Vec] | None = None,
    constraint_multipliers: np.ndarray | None = None,
) -> PETSc.Vec:
    """Assemble the physical finite-element equilibrium residual.

    The unconstrained finite-element residual is

        r = K u - f.

    Global rigid-body Lagrange-multiplier forces are then added so that the
    returned vector represents the physical residual associated with the
    Dirichlet-constrained problem.

    Parameters
    ----------
    formulation
        Finite-element formulation providing the residual form.
    displacement
        Solved displacement field.
    constraint_rows
        Assembled displacement-space constraint rows.
    constraint_multipliers
        Solved Lagrange multipliers associated with ``constraint_rows``.

    Returns
    -------
    PETSc.Vec
        Physical displacement-space residual vector.
    """
    residual_form = fem.form(
        formulation.residual_form(displacement),
    )

    residual = petsc.assemble_vector(residual_form)

    residual.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )

    if constraint_rows and constraint_multipliers is not None:
        if len(constraint_rows) != len(constraint_multipliers):
            raise ValueError(
                "Number of constraint rows must match number of "
                "constraint multipliers.",
            )

        for row, multiplier in zip(
            constraint_rows,
            constraint_multipliers,
        ):
            residual.axpy(
                float(multiplier),
                row,
            )

        residual.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )

    return residual


def compute_discrete_reaction(
    formulation: Any,
    displacement: fem.Function,
    mesh_data: Any,
    boundary: Any,
    boundary_name: str,
    constraint_rows: list[PETSc.Vec] | None = None,
    constraint_multipliers: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the resultant reaction on a named Dirichlet boundary.

    The reaction is obtained from the physical equilibrium residual evaluated
    at the prescribed displacement degrees of freedom associated with the
    requested physical boundary.

    The returned vector contains one global resultant force component for
    each spatial direction.

    Parameters
    ----------
    formulation
        Finite-element formulation providing the residual form.
    displacement
        Solved displacement field.
    mesh_data
        Mesh data associated with the finite-element problem.
    boundary
        Problem-level boundary definitions.
    boundary_name
        Name of the physical boundary on which the reaction is evaluated.
    constraint_rows
        Assembled displacement-space constraint rows.
    constraint_multipliers
        Solved Lagrange multipliers associated with ``constraint_rows``.

    Returns
    -------
    numpy.ndarray
        Three-component global reaction-force vector.
    """
    residual = assemble_physical_residual(
        formulation=formulation,
        displacement=displacement,
        constraint_rows=constraint_rows,
        constraint_multipliers=constraint_multipliers,
    )

    located_dofs = locate_dirichlet_dofs(
        mesh_data=mesh_data,
        function_space=formulation.function_space,
        boundary=boundary,
        boundary_name=boundary_name,
    )

    reaction = np.zeros(
        formulation.mesh.geometry.dim,
        dtype=float,
    )

    for component, dofs in located_dofs:
        component_index = {"x": 0, "y": 1, "z": 2}[component]

        subspace = formulation.function_space.sub(component_index)
        owned_size = subspace.dofmap.index_map.size_local

        owned_dofs = dofs[dofs < owned_size]

        if owned_dofs.size == 0:
            continue

        local_values = residual.getValues(
            owned_dofs.tolist(),
        )

        reaction[component_index] += np.sum(local_values)

    formulation.mesh.comm.Allreduce(
        MPI.IN_PLACE,
        reaction,
        op=MPI.SUM,
    )

    return reaction
