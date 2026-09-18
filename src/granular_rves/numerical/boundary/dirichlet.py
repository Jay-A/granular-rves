"""Construction of numerical Dirichlet boundary conditions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dolfinx import default_scalar_type, fem

from granular_rves.problem.definition import BoundaryDefinition


_COMPONENT_INDICES = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def create_dirichlet_bcs(
    mesh_data: Any,
    function_space: fem.FunctionSpace,
    boundary: BoundaryDefinition,
    values: Mapping[str, float] | None = None,
) -> list[Any]:
    """Create DOLFINx Dirichlet boundary conditions.

    Problem-level Dirichlet boundary definitions identify named geometric
    regions and constrained displacement components. This function translates
    those definitions into DOLFINx boundary-condition objects using the
    physical groups and facet tags stored in ``mesh_data``.

    Parameters
    ----------
    mesh_data
        DOLFINx ``MeshData`` returned by the mesh-generation layer. It must
        provide ``mesh``, ``facet_tags``, and ``physical_groups``.
    function_space
        Vector-valued DOLFINx function space on ``mesh_data.mesh``.
    boundary
        Problem-level Dirichlet boundary definitions.
    values
        Current values for loading-controlled constraints. The keys are
        boundary region names. A value is required only for constraints whose
        definition has ``value=None``.

    Returns
    -------
    list
        DOLFINx Dirichlet boundary-condition objects.

    Raises
    ------
    ValueError
        If a boundary region does not exist in the mesh, a boundary component
        is invalid for the mesh dimension, or a loading-controlled boundary
        has no current value.
    """
    if values is None:
        values = {}

    mesh = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    physical_groups = mesh_data.physical_groups

    bcs: list[Any] = []

    for region, definition in boundary.dirichlet.items():
        try:
            physical_group = physical_groups[region]
        except KeyError as exc:
            raise ValueError(
                f"Dirichlet boundary region {region!r} is not present "
                "in the mesh physical groups."
            ) from exc

        if physical_group.dim != mesh.topology.dim - 1:
            raise ValueError(
                f"Dirichlet boundary region {region!r} has geometric "
                f"dimension {physical_group.dim}, but boundary facets "
                f"have dimension {mesh.topology.dim - 1}."
            )

        try:
            component = _COMPONENT_INDICES[definition.component]
        except KeyError as exc:
            raise ValueError(
                f"Unknown displacement component "
                f"{definition.component!r}. Expected one of "
                f"{tuple(_COMPONENT_INDICES)}."
            ) from exc

        if component >= mesh.geometry.dim:
            raise ValueError(
                f"Displacement component {definition.component!r} is not "
                f"available on a {mesh.geometry.dim}D mesh."
            )

        if definition.value is None:
            try:
                value = values[region]
            except KeyError as exc:
                raise ValueError(
                    f"No current value was supplied for loading-controlled "
                    f"Dirichlet boundary {region!r}."
                ) from exc
        else:
            value = definition.value

        facets = facet_tags.find(physical_group.tag)

        if facets.size == 0:
            raise ValueError(
                f"Dirichlet boundary region {region!r} has no tagged facets."
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
