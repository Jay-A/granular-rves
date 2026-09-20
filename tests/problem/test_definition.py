"""Tests for declarative simulation problem definitions."""

from __future__ import annotations

import pytest

from granular_rves.mechanics.rigid_body import RigidBodyConstraintMode
from granular_rves.problem.definition import (
    BalanceDefinition,
    BoundaryConditionDefinition,
    BoundaryDefinition,
    ConstitutiveDefinition,
    DirichletBoundaryDefinition,
    GeometryDefinition,
    KinematicsDefinition,
    LoadingDefinition,
    MechanicsDefinition,
    MeshDefinition,
    OutputDefinition,
    ProblemDefinition,
    RigidBodyConstraintDefinition,
)


def make_boundary() -> BoundaryConditionDefinition:
    """Construct representative problem boundaries and conditions."""
    return BoundaryConditionDefinition(
        boundaries={
            "bottom": BoundaryDefinition(face="bottom"),
            "top": BoundaryDefinition(face="top"),
            "lateral": BoundaryDefinition(face="lateral"),
        },
        dirichlet={
            "bottom": DirichletBoundaryDefinition(
                component="z",
                value=0.0,
            ),
            "top": DirichletBoundaryDefinition(
                component="z",
                value=None,
            ),
        },
    )


def make_mechanics() -> MechanicsDefinition:
    """Construct representative mechanics definitions."""
    return MechanicsDefinition(
        kinematics=KinematicsDefinition(
            models={
                "small_strain": {},
            },
        ),
        constitutive=ConstitutiveDefinition(
            models={
                "linear_elastic": {
                    "youngs_modulus": 1.0e6,
                    "poisson_ratio": 0.3,
                },
            },
        ),
        balance=BalanceDefinition(
            models={
                "momentum": {},
            },
        ),
    )


def make_problem() -> ProblemDefinition:
    """Construct a representative steady cylinder problem."""
    return ProblemDefinition(
        name="cylinder_compression",
        analysis="steady",
        geometry=GeometryDefinition(
            type="cylinder",
            parameters={
                "x0": [0.0, 0.0, 0.0],
                "x1": [0.0, 0.0, 1.0],
                "radius": 1.0,
            },
        ),
        mesh=MeshDefinition(
            size=0.25,
        ),
        mechanics=make_mechanics(),
        boundary=make_boundary(),
        constraints=RigidBodyConstraintDefinition(
            reference_boundary="bottom",
            translation_x=RigidBodyConstraintMode.MEAN_ZERO,
            translation_y=RigidBodyConstraintMode.MEAN_ZERO,
            rotation_z=RigidBodyConstraintMode.MEAN_ZERO,
        ),
        loading=LoadingDefinition(
            type="displacement",
            boundary="top",
            component="z",
            value=-0.01,
        ),
        output=OutputDefinition(
            directory="output/cylinder_compression",
        ),
    )


def test_dirichlet_boundary_definition() -> None:
    """Represent fixed and loading-controlled Dirichlet conditions."""
    fixed = DirichletBoundaryDefinition(
        component="z",
        value=0.0,
    )

    loading_controlled = DirichletBoundaryDefinition(
        component="z",
        value=None,
    )

    assert fixed.component == "z"
    assert fixed.value == 0.0

    assert loading_controlled.component == "z"
    assert loading_controlled.value is None


def test_boundary_definition() -> None:
    """Store generic boundaries separately from Dirichlet conditions."""
    boundary = make_boundary()

    assert set(boundary.boundaries) == {
        "bottom",
        "top",
        "lateral",
    }

    assert boundary.boundaries["bottom"].face == "bottom"
    assert boundary.boundaries["top"].face == "top"
    assert boundary.boundaries["lateral"].face == "lateral"

    assert set(boundary.dirichlet) == {
        "bottom",
        "top",
    }

    assert boundary.dirichlet["bottom"].component == "z"
    assert boundary.dirichlet["bottom"].value == 0.0

    assert boundary.dirichlet["top"].component == "z"
    assert boundary.dirichlet["top"].value is None


def test_problem_definition() -> None:
    """Compose the expected simulation problem components."""
    problem = make_problem()

    assert problem.name == "cylinder_compression"
    assert problem.analysis == "steady"

    assert isinstance(problem.geometry, GeometryDefinition)
    assert problem.geometry.type == "cylinder"
    assert problem.geometry.parameters == {
        "x0": [0.0, 0.0, 0.0],
        "x1": [0.0, 0.0, 1.0],
        "radius": 1.0,
    }

    assert problem.mesh.size == 0.25
    assert problem.mesh.order == 1

    assert isinstance(problem.mechanics, MechanicsDefinition)

    assert problem.mechanics.kinematics.models == {
        "small_strain": {},
    }

    assert problem.mechanics.constitutive.models == {
        "linear_elastic": {
            "youngs_modulus": 1.0e6,
            "poisson_ratio": 0.3,
        },
    }

    assert problem.mechanics.balance.models == {
        "momentum": {},
    }

    assert isinstance(
        problem.boundary,
        BoundaryConditionDefinition,
    )

    assert problem.boundary.boundaries["bottom"].face == "bottom"
    assert problem.boundary.boundaries["top"].face == "top"

    assert problem.boundary.dirichlet["bottom"].value == 0.0
    assert problem.boundary.dirichlet["top"].value is None

    assert problem.constraints.reference_boundary == "bottom"
    assert (
        problem.constraints.translation_x
        is RigidBodyConstraintMode.MEAN_ZERO
    )
    assert (
        problem.constraints.translation_y
        is RigidBodyConstraintMode.MEAN_ZERO
    )
    assert (
        problem.constraints.rotation_z
        is RigidBodyConstraintMode.MEAN_ZERO
    )

    assert problem.loading.type == "displacement"
    assert problem.loading.boundary == "top"
    assert problem.loading.component == "z"
    assert problem.loading.value == -0.01

    assert problem.output.directory == "output/cylinder_compression"


def test_definitions_are_immutable() -> None:
    """Problem definitions cannot be modified after construction."""
    problem = make_problem()

    with pytest.raises(AttributeError):
        problem.name = "modified"


def test_geometry_definition_is_preserved() -> None:
    """Problem definition preserves the supplied geometry definition."""
    geometry = GeometryDefinition(
        type="cylinder",
        parameters={
            "x0": [0.0, 0.0, 0.0],
            "x1": [0.0, 0.0, 2.0],
            "radius": 1.0,
        },
    )

    problem = ProblemDefinition(
        name="test",
        analysis="steady",
        geometry=geometry,
        mesh=MeshDefinition(
            size=0.25,
        ),
        mechanics=make_mechanics(),
        boundary=make_boundary(),
        loading=LoadingDefinition(
            type="displacement",
            boundary="top",
            component="z",
            value=-0.01,
        ),
        output=OutputDefinition(
            directory="output/test",
        ),
    )

    assert problem.geometry is geometry


def test_rigid_body_constraint_definition_defaults() -> None:
    """Rigid-body constraints default to unconstrained."""
    constraints = RigidBodyConstraintDefinition()

    assert (
        constraints.translation_x
        is RigidBodyConstraintMode.UNCONSTRAINED
    )
    assert (
        constraints.translation_y
        is RigidBodyConstraintMode.UNCONSTRAINED
    )
    assert (
        constraints.translation_z
        is RigidBodyConstraintMode.UNCONSTRAINED
    )
    assert (
        constraints.rotation_x
        is RigidBodyConstraintMode.UNCONSTRAINED
    )
    assert (
        constraints.rotation_y
        is RigidBodyConstraintMode.UNCONSTRAINED
    )
    assert (
        constraints.rotation_z
        is RigidBodyConstraintMode.UNCONSTRAINED
    )


def test_rigid_body_constraint_definition_preserves_configuration() -> None:
    """Rigid-body constraints preserve their configured modes."""
    constraints = RigidBodyConstraintDefinition(
        reference_boundary="bottom",
        translation_x=RigidBodyConstraintMode.MEAN_ZERO,
        translation_y=RigidBodyConstraintMode.MEAN_ZERO,
        translation_z=RigidBodyConstraintMode.ZERO,
        rotation_x=RigidBodyConstraintMode.UNCONSTRAINED,
        rotation_y=RigidBodyConstraintMode.UNCONSTRAINED,
        rotation_z=RigidBodyConstraintMode.ZERO,
    )

    assert constraints.reference_boundary == "bottom"

    assert (
        constraints.translation_x
        is RigidBodyConstraintMode.MEAN_ZERO
    )
    assert (
        constraints.translation_y
        is RigidBodyConstraintMode.MEAN_ZERO
    )
    assert (
        constraints.translation_z
        is RigidBodyConstraintMode.ZERO
    )
    assert (
        constraints.rotation_x
        is RigidBodyConstraintMode.UNCONSTRAINED
    )
    assert (
        constraints.rotation_y
        is RigidBodyConstraintMode.UNCONSTRAINED
    )
    assert (
        constraints.rotation_z
        is RigidBodyConstraintMode.ZERO
    )


def test_problem_definition_defaults_rigid_body_constraints() -> None:
    """Problem definitions default rigid-body constraints to unconstrained."""
    problem = ProblemDefinition(
        name="test",
        analysis="steady",
        geometry=GeometryDefinition(
            type="cylinder",
            parameters={},
        ),
        mesh=MeshDefinition(
            size=0.25,
        ),
        mechanics=make_mechanics(),
        boundary=make_boundary(),
    )

    assert isinstance(
        problem.constraints,
        RigidBodyConstraintDefinition,
    )

    assert (
        problem.constraints.translation_x
        is RigidBodyConstraintMode.UNCONSTRAINED
    )
    assert (
        problem.constraints.translation_y
        is RigidBodyConstraintMode.UNCONSTRAINED
    )
    assert (
        problem.constraints.translation_z
        is RigidBodyConstraintMode.UNCONSTRAINED
    )
    assert (
        problem.constraints.rotation_x
        is RigidBodyConstraintMode.UNCONSTRAINED
    )
    assert (
        problem.constraints.rotation_y
        is RigidBodyConstraintMode.UNCONSTRAINED
    )
    assert (
        problem.constraints.rotation_z
        is RigidBodyConstraintMode.UNCONSTRAINED
    )


