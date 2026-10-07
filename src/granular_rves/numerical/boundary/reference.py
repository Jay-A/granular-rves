"""Construction of UFL measures on named problem boundaries.

This module resolves problem-level boundary definitions through the mesh
physical groups and constructs UFL boundary measures for numerical
formulations that require integration over a specific boundary.
"""

from __future__ import annotations

from typing import Any

import ufl

from granular_rves.problem.definition import BoundaryConditionDefinition


def create_boundary_measure(
    mesh_data: Any,
    boundary: BoundaryConditionDefinition,
    boundary_name: str,
) -> ufl.Measure:
    """Create a UFL boundary measure for a named problem boundary.

    The problem boundary name is resolved through the generic boundary
    registry to a geometry face. The geometry face is then resolved
    through the physical groups and facet tags stored in ``mesh_data``.

    Parameters
    ----------
    mesh_data
        DOLFINx ``MeshData`` returned by the mesh-generation layer. It
        must provide ``mesh``, ``facet_tags``, and ``physical_groups``.
    boundary
        Problem-level boundary definitions.
    boundary_name
        Name of the problem boundary on which the measure is defined.

    Returns
    -------
    ufl.Measure
        Boundary measure restricted to the facets belonging to the
        requested problem boundary.

    Raises
    ------
    ValueError
        If the requested problem boundary is not declared, its geometry
        face is not present in the mesh physical groups, the physical
        group does not describe boundary facets, or no tagged facets
        are associated with the geometry face.

    Examples
    --------
    Construct a measure over the configured reference boundary::

        reference_measure = create_boundary_measure(
            mesh_data,
            problem.boundary,
            problem.constraints.reference_boundary,
        )
    """
    try:
        problem_boundary = boundary.boundaries[boundary_name]
    except KeyError as exc:
        raise ValueError(
            f"Problem boundary {boundary_name!r} is not declared "
            "in the boundary registry."
        ) from exc

    geometry_face = problem_boundary.face

    mesh = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    physical_groups = mesh_data.physical_groups

    try:
        physical_group = physical_groups[geometry_face]
    except KeyError as exc:
        raise ValueError(
            f"Geometry face {geometry_face!r}, associated with "
            f"problem boundary {boundary_name!r}, is not present "
            "in the mesh physical groups."
        ) from exc

    expected_dimension = mesh.topology.dim - 1

    if physical_group.dim != expected_dimension:
        raise ValueError(
            f"Geometry face {geometry_face!r}, associated with "
            f"problem boundary {boundary_name!r}, has geometric "
            f"dimension {physical_group.dim}, but boundary facets "
            f"have dimension {expected_dimension}."
        )

    facets = facet_tags.find(physical_group.tag)

    if facets.size == 0:
        raise ValueError(
            f"Geometry face {geometry_face!r}, associated with "
            f"problem boundary {boundary_name!r}, has no tagged "
            "facets."
        )

    return ufl.Measure(
        "ds",
        domain=mesh,
        subdomain_data=facet_tags,
        subdomain_id=physical_group.tag,
    )


