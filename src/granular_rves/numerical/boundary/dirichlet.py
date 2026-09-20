"""Construction of numerical Dirichlet boundary conditions.

This module translates problem-level Dirichlet boundary definitions into
DOLFINx boundary-condition objects. It resolves named problem boundaries
through the generic boundary registry and the physical groups produced by
the mesh-generation layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dolfinx import default_scalar_type, fem

from granular_rves.problem.definition import BoundaryConditionDefinition


_COMPONENT_INDICES = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def create_dirichlet_bcs(
    mesh_data: Any,
    function_space: fem.FunctionSpace,
    boundary: BoundaryConditionDefinition,
    values: Mapping[str, float] | None = None,
) -> list[Any]:
    """Create DOLFINx Dirichlet boundary conditions.

    Problem-level Dirichlet definitions identify named problem boundaries
    and displacement components. The named problem boundaries are resolved
    through the generic boundary registry to geometry faces, which are then
    resolved through the mesh physical groups and facet tags.

    Parameters
    ----------
    mesh_data
        DOLFINx ``MeshData`` returned by the mesh-generation layer. It
        must provide ``mesh``, ``facet_tags``, and ``physical_groups``.
    function_space
        Vector-valued DOLFINx function space on ``mesh_data.mesh``.
    boundary
        Problem-level boundary and Dirichlet-condition definitions.
    values
        Current values for loading-controlled constraints. Keys are
        problem boundary names. A value is required only when the
        corresponding Dirichlet definition has ``value=None``.

    Returns
    -------
    list
        DOLFINx Dirichlet boundary-condition objects.

    Raises
    ------
    ValueError
        If a referenced problem boundary or geometry face is missing,
        a physical group has the wrong dimension, a displacement
        component is invalid, a loading-controlled value is missing,
        or no facets are associated with a boundary.
    """
    if values is None:
        values = {}

    mesh = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    physical_groups = mesh_data.physical_groups

    bcs: list[Any] = []

    for boundary_name, definition in boundary.dirichlet.items():
        try:
            problem_boundary = boundary.boundaries[boundary_name]
        except KeyError as exc:
            raise ValueError(
                f"Dirichlet boundary {boundary_name!r} is not declared "
                "in the problem boundary registry."
            ) from exc

        geometry_face = problem_boundary.face

        try:
            physical_group = physical_groups[geometry_face]
        except KeyError as exc:
            raise ValueError(
                f"Geometry face {geometry_face!r}, associated with "
                f"Dirichlet boundary {boundary_name!r}, is not present "
                "in the mesh physical groups."
            ) from exc

        expected_dimension = mesh.topology.dim - 1

        if physical_group.dim != expected_dimension:
            raise ValueError(
                f"Geometry face {geometry_face!r}, associated with "
                f"Dirichlet boundary {boundary_name!r}, has geometric "
                f"dimension {physical_group.dim}, but boundary facets "
                f"have dimension {expected_dimension}."
            )

        try:
            component = _COMPONENT_INDICES[definition.component]
        except KeyError as exc:
            raise ValueError(
                f"Unknown displacement component "
                f"{definition.component!r} for Dirichlet boundary "
                f"{boundary_name!r}. Expected one of "
                f"{tuple(_COMPONENT_INDICES)}."
            ) from exc

        if component >= mesh.geometry.dim:
            raise ValueError(
                f"Displacement component {definition.component!r} is "
                f"not available on a {mesh.geometry.dim}D mesh."
            )

        if definition.value is None:
            try:
                value = values[boundary_name]
            except KeyError as exc:
                raise ValueError(
                    "No current value was supplied for "
                    f"loading-controlled Dirichlet boundary "
                    f"{boundary_name!r}."
                ) from exc
        else:
            value = definition.value

        facets = facet_tags.find(physical_group.tag)

        if facets.size == 0:
            raise ValueError(
                f"Geometry face {geometry_face!r}, associated with "
                f"Dirichlet boundary {boundary_name!r}, has no tagged "
                "facets."
            )

        subspace = function_space.sub(component)

        dofs = fem.locate_dofs_topological(
            subspace,
            mesh.topology.dim - 1,
            facets,
        )

        value_function = fem.Constant(
            mesh,
            default_scalar_type(value),
        )

        bcs.append(
            fem.dirichletbc(
                value_function,
                dofs,
                subspace,
            )
        )

    return bcs


