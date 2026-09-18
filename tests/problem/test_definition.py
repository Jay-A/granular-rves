from __future__ import annotations

import pytest

from granular_rves.mechanics.definitions import (
    BalanceDefinition,
    ConstitutiveDefinition,
    KinematicsDefinition,
    MechanicsDefinition,
)
from granular_rves.problem.definition import (
    AnalysisDefinition,
    BoundaryDefinition,
    DirichletBoundaryDefinition,
    LoadingDefinition,
    MeshDefinition,
    OutputDefinition,
    ProblemDefinition,
)
from granular_rves.problem.geometry.geometry_types.cylinder import Cylinder


def make_boundary() -> BoundaryDefinition:
    """Construct representative Dirichlet boundary constraints."""
    return BoundaryDefinition(
        dirichlet={
            "bottom": DirichletBoundaryDefinition(
                region="bottom",
                component="z",
                value=0.0,
            ),
            "top": DirichletBoundaryDefinition(
                region="top",
                component="z",
                value=None,
            ),
        },
    )


def make_problem() -> ProblemDefinition:
    """Construct a representative quasi-static cylinder problem."""
    return ProblemDefinition(
        name="cylinder_compression",
        analysis=AnalysisDefinition(
            type="quasi_static",
        ),
        geometry=Cylinder(
            x0=(0.0, 0.0, 0.0),
            x1=(0.0, 0.0, 2.0),
            radius=1.0,
        ),
        mesh=MeshDefinition(
            size=0.25,
        ),
        mechanics=MechanicsDefinition(
            kinematics=KinematicsDefinition(
                type="small_strain",
                parameters={},
            ),
            constitutive=ConstitutiveDefinition(
                type="linear_elastic",
                parameters={
                    "youngs_modulus": 1.0e6,
                    "poisson_ratio": 0.3,
                },
            ),
            balance=BalanceDefinition(
                type="momentum",
                parameters={},
            ),
        ),
        boundary=make_boundary(),
        loading=LoadingDefinition(
            type="displacement",
            region="top",
            component="z",
            start=0.0,
            end=-0.1,
            steps=100,
        ),
        output=OutputDefinition(
            directory="output/cylinder_compression",
        ),
    )


def test_dirichlet_boundary_definition() -> None:
    """Represent fixed and loading-controlled Dirichlet constraints."""
    fixed = DirichletBoundaryDefinition(
        region="bottom",
        component="z",
        value=0.0,
    )

    loading_controlled = DirichletBoundaryDefinition(
        region="top",
        component="z",
        value=None,
    )

    assert fixed.region == "bottom"
    assert fixed.component == "z"
    assert fixed.value == 0.0

    assert loading_controlled.region == "top"
    assert loading_controlled.component == "z"
    assert loading_controlled.value is None


def test_boundary_definition() -> None:
    """Boundary definition stores the configured Dirichlet constraints."""
    boundary = make_boundary()

    assert set(boundary.dirichlet) == {"bottom", "top"}

    assert boundary.dirichlet["bottom"].region == "bottom"
    assert boundary.dirichlet["bottom"].component == "z"
    assert boundary.dirichlet["bottom"].value == 0.0

    assert boundary.dirichlet["top"].region == "top"
    assert boundary.dirichlet["top"].component == "z"
    assert boundary.dirichlet["top"].value is None


def test_problem_definition() -> None:
    """Problem definition composes the expected simulation components."""
    problem = make_problem()

    assert problem.name == "cylinder_compression"
    assert problem.analysis.type == "quasi_static"
    assert isinstance(problem.geometry, Cylinder)
    assert problem.mesh.size == 0.25

    assert isinstance(problem.mechanics, MechanicsDefinition)
    assert problem.mechanics.kinematics.type == "small_strain"
    assert problem.mechanics.constitutive.type == "linear_elastic"
    assert problem.mechanics.balance.type == "momentum"

    assert isinstance(problem.boundary, BoundaryDefinition)
    assert problem.boundary.dirichlet["bottom"].value == 0.0
    assert problem.boundary.dirichlet["top"].value is None

    assert problem.loading.type == "displacement"
    assert problem.loading.region == "top"
    assert problem.loading.component == "z"
    assert problem.loading.start == 0.0
    assert problem.loading.end == -0.1
    assert problem.loading.steps == 100

    assert problem.output.directory == "output/cylinder_compression"


def test_definitions_are_immutable() -> None:
    """Problem definitions cannot be modified after construction."""
    problem = make_problem()

    with pytest.raises(AttributeError):
        problem.name = "modified"


def test_geometry_is_preserved() -> None:
    """Problem definition preserves the supplied geometry object."""
    geometry = Cylinder(
        x0=(0.0, 0.0, 0.0),
        x1=(0.0, 0.0, 2.0),
        radius=1.0,
    )

    problem = ProblemDefinition(
        name="test",
        analysis=AnalysisDefinition(
            type="quasi_static",
        ),
        geometry=geometry,
        mesh=MeshDefinition(
            size=0.25,
        ),
        mechanics=MechanicsDefinition(
            kinematics=KinematicsDefinition(
                type="small_strain",
                parameters={},
            ),
            constitutive=ConstitutiveDefinition(
                type="linear_elastic",
                parameters={
                    "youngs_modulus": 1.0e6,
                    "poisson_ratio": 0.3,
                },
            ),
            balance=BalanceDefinition(
                type="momentum",
                parameters={},
            ),
        ),
        boundary=make_boundary(),
        loading=LoadingDefinition(
            type="displacement",
            region="top",
            component="z",
            start=0.0,
            end=-0.1,
            steps=10,
        ),
        output=OutputDefinition(
            directory="output/test",
        ),
    )

    assert problem.geometry is geometry


