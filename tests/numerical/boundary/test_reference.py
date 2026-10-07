"""Tests for numerical reference-boundary measures."""

from __future__ import annotations

import pytest
from mpi4py import MPI

from granular_rves.numerical.boundary.reference import (
    create_boundary_measure,
)
from granular_rves.problem.definition import (
    BoundaryConditionDefinition,
    BoundaryDefinition,
)
from granular_rves.problem.geometry.geometry_types.cylinder import Cylinder
from granular_rves.problem.geometry.meshing import (
    MeshSettings,
    create_mesh,
)


def make_mesh_data():
    """Create a representative cylinder mesh."""
    geometry = Cylinder(
        x0=(0.0, 0.0, 0.0),
        x1=(0.0, 0.0, 1.0),
        radius=1.0,
    )

    return create_mesh(
        geometry=[geometry],
        settings=MeshSettings(characteristic_length=0.5),
        comm=MPI.COMM_SELF,
    )


def make_boundary_definition():
    """Create the generic problem-boundary registry."""
    return BoundaryConditionDefinition(
        boundaries={
            "bottom": BoundaryDefinition(face="bottom"),
            "top": BoundaryDefinition(face="top"),
            "lateral": BoundaryDefinition(face="lateral"),
        },
    )


def test_create_boundary_measure() -> None:
    """Create a UFL boundary measure for a named problem boundary."""
    mesh_data = make_mesh_data()
    boundary = make_boundary_definition()

    measure = create_boundary_measure(
        mesh_data=mesh_data,
        boundary=boundary,
        boundary_name="bottom",
    )

    assert measure is not None


@pytest.mark.parametrize(
    "boundary_name",
    ["bottom", "top", "lateral"],
)
def test_create_boundary_measure_for_supported_boundaries(
    boundary_name: str,
) -> None:
    """Create measures for each declared cylinder boundary."""
    mesh_data = make_mesh_data()
    boundary = make_boundary_definition()

    measure = create_boundary_measure(
        mesh_data=mesh_data,
        boundary=boundary,
        boundary_name=boundary_name,
    )

    assert measure is not None


def test_unknown_problem_boundary_is_rejected() -> None:
    """Reject a boundary name that is not declared in the problem."""
    mesh_data = make_mesh_data()
    boundary = make_boundary_definition()

    with pytest.raises(ValueError, match="missing"):
        create_boundary_measure(
            mesh_data=mesh_data,
            boundary=boundary,
            boundary_name="missing",
        )


def test_unknown_geometry_face_is_rejected() -> None:
    """Reject a problem boundary referring to an unknown geometry face."""
    mesh_data = make_mesh_data()

    boundary = BoundaryConditionDefinition(
        boundaries={
            "reference": BoundaryDefinition(face="missing"),
        },
    )

    with pytest.raises(
        ValueError,
        match="not present in the mesh physical groups",
    ):
        create_boundary_measure(
            mesh_data=mesh_data,
            boundary=boundary,
            boundary_name="reference",
        )


