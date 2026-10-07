from __future__ import annotations

import pytest
from dolfinx import fem
from mpi4py import MPI

from granular_rves.numerical.boundary.dirichlet import create_dirichlet_bcs
from granular_rves.problem.definition import (
    BoundaryConditionDefinition,
    BoundaryDefinition,
    DirichletBoundaryDefinition,
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


def make_function_space(mesh_data):
    """Create the displacement function space used by the formulation."""
    return fem.functionspace(
        mesh_data.mesh,
        ("Lagrange", 1, (mesh_data.mesh.geometry.dim,)),
    )


def make_boundary_definition(*dirichlet):
    """Create the generic boundary registry and Dirichlet definitions."""
    boundaries = {
        "bottom": BoundaryDefinition(face="bottom"),
        "top": BoundaryDefinition(face="top"),
        "lateral": BoundaryDefinition(face="lateral"),
    }

    return BoundaryConditionDefinition(
        boundaries=boundaries,
        dirichlet=dict(dirichlet),
    )


def test_create_fixed_dirichlet_bcs() -> None:
    """Create a Dirichlet BC from a fixed boundary definition."""
    mesh_data = make_mesh_data()
    function_space = make_function_space(mesh_data)

    boundary = make_boundary_definition(
        (
            "bottom",
            DirichletBoundaryDefinition(
                component="z",
                value=0.0,
            ),
        ),
    )

    bcs = create_dirichlet_bcs(
        mesh_data=mesh_data,
        function_space=function_space,
        boundary=boundary,
    )

    assert len(bcs) == 1


def test_create_loading_controlled_dirichlet_bc() -> None:
    """Create a Dirichlet BC using a supplied current loading value."""
    mesh_data = make_mesh_data()
    function_space = make_function_space(mesh_data)

    boundary = make_boundary_definition(
        (
            "top",
            DirichletBoundaryDefinition(
                component="z",
                value=None,
            ),
        ),
    )

    bcs = create_dirichlet_bcs(
        mesh_data=mesh_data,
        function_space=function_space,
        boundary=boundary,
        values={"top": -0.004},
    )

    assert len(bcs) == 1


def test_create_multiple_dirichlet_bcs() -> None:
    """Create fixed and loading-controlled Dirichlet BCs together."""
    mesh_data = make_mesh_data()
    function_space = make_function_space(mesh_data)

    boundary = make_boundary_definition(
        (
            "bottom",
            DirichletBoundaryDefinition(
                component="z",
                value=0.0,
            ),
        ),
        (
            "top",
            DirichletBoundaryDefinition(
                component="z",
                value=None,
            ),
        ),
    )

    bcs = create_dirichlet_bcs(
        mesh_data=mesh_data,
        function_space=function_space,
        boundary=boundary,
        values={"top": -0.004},
    )

    assert len(bcs) == 2


def test_loading_controlled_boundary_requires_current_value() -> None:
    """Reject a loading-controlled boundary without a current value."""
    mesh_data = make_mesh_data()
    function_space = make_function_space(mesh_data)

    boundary = make_boundary_definition(
        (
            "top",
            DirichletBoundaryDefinition(
                component="z",
                value=None,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="No current value was supplied",
    ):
        create_dirichlet_bcs(
            mesh_data=mesh_data,
            function_space=function_space,
            boundary=boundary,
        )


def test_unknown_problem_boundary_is_rejected() -> None:
    """Reject a Dirichlet condition for an undeclared problem boundary."""
    mesh_data = make_mesh_data()
    function_space = make_function_space(mesh_data)

    boundary = BoundaryConditionDefinition(
        boundaries={
            "bottom": BoundaryDefinition(face="bottom"),
            "top": BoundaryDefinition(face="top"),
            "lateral": BoundaryDefinition(face="lateral"),
        },
        dirichlet={
            "missing": DirichletBoundaryDefinition(
                component="z",
                value=0.0,
            ),
        },
    )

    with pytest.raises(ValueError, match="missing"):
        create_dirichlet_bcs(
            mesh_data=mesh_data,
            function_space=function_space,
            boundary=boundary,
        )


def test_unknown_geometry_face_is_rejected() -> None:
    """Reject a problem boundary referring to an unknown geometry face."""
    mesh_data = make_mesh_data()
    function_space = make_function_space(mesh_data)

    boundary = BoundaryConditionDefinition(
        boundaries={
            "missing": BoundaryDefinition(face="missing"),
        },
        dirichlet={
            "missing": DirichletBoundaryDefinition(
                component="z",
                value=0.0,
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="not present in the mesh physical groups",
    ):
        create_dirichlet_bcs(
            mesh_data=mesh_data,
            function_space=function_space,
            boundary=boundary,
        )


@pytest.mark.parametrize("component", ["x", "y", "z"])
def test_supported_displacement_components(component: str) -> None:
    """Create constraints for each Cartesian displacement component."""
    mesh_data = make_mesh_data()
    function_space = make_function_space(mesh_data)

    boundary = make_boundary_definition(
        (
            "top",
            DirichletBoundaryDefinition(
                component=component,
                value=0.0,
            ),
        ),
    )

    bcs = create_dirichlet_bcs(
        mesh_data=mesh_data,
        function_space=function_space,
        boundary=boundary,
    )

    assert len(bcs) == 1


def test_unknown_displacement_component_is_rejected() -> None:
    """Reject an unsupported displacement component."""
    mesh_data = make_mesh_data()
    function_space = make_function_space(mesh_data)

    boundary = make_boundary_definition(
        (
            "top",
            DirichletBoundaryDefinition(
                component="invalid",
                value=0.0,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Unknown displacement component",
    ):
        create_dirichlet_bcs(
            mesh_data=mesh_data,
            function_space=function_space,
            boundary=boundary,
        )


