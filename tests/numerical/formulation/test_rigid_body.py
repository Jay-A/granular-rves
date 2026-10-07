"""Tests for variational rigid-body reference constraints."""

from __future__ import annotations

import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from granular_rves.mechanics.rigid_body import (
    RigidBodyConstraintMode,
)
from granular_rves.numerical.formulation.rigid_body import (
    RigidBodyFormulation,
)
from granular_rves.problem.definition import RigidBodyConstraintDefinition


@pytest.fixture
def function_space():
    """Create a vector-valued first-order Lagrange space."""
    domain = mesh.create_unit_cube(
        MPI.COMM_WORLD,
        2,
        2,
        2,
        mesh.CellType.tetrahedron,
    )

    return fem.functionspace(
        domain,
        ("Lagrange", 1, (domain.geometry.dim,)),
    )


@pytest.fixture
def reference_measure(function_space):
    """Create a boundary measure for formulation tests."""
    return ufl.Measure(
        "ds",
        domain=function_space.mesh,
    )


@pytest.fixture
def formulation(function_space, reference_measure):
    """Create a rigid-body formulation."""
    return RigidBodyFormulation(
        function_space=function_space,
        reference_measure=reference_measure,
    )


def test_translation_x_returns_form(formulation):
    """Translation-x constraint returns a compilable UFL form."""
    form = formulation.translation_x()

    assert isinstance(form, ufl.Form)
    assert len(form.integrals()) == 1
    assert fem.form(form) is not None


def test_translation_y_returns_form(formulation):
    """Translation-y constraint returns a compilable UFL form."""
    form = formulation.translation_y()

    assert isinstance(form, ufl.Form)
    assert len(form.integrals()) == 1
    assert fem.form(form) is not None


def test_translation_z_returns_form(formulation):
    """Translation-z constraint returns a compilable UFL form."""
    form = formulation.translation_z()

    assert isinstance(form, ufl.Form)
    assert len(form.integrals()) == 1
    assert fem.form(form) is not None


def test_rotation_x_returns_form(formulation):
    """Rotation-x constraint returns a compilable UFL form."""
    form = formulation.rotation_x()

    assert isinstance(form, ufl.Form)
    assert len(form.integrals()) == 1
    assert fem.form(form) is not None


def test_rotation_y_returns_form(formulation):
    """Rotation-y constraint returns a compilable UFL form."""
    form = formulation.rotation_y()

    assert isinstance(form, ufl.Form)
    assert len(form.integrals()) == 1
    assert fem.form(form) is not None


def test_rotation_z_returns_form(formulation):
    """Rotation-z constraint returns a compilable UFL form."""
    form = formulation.rotation_z()

    assert isinstance(form, ufl.Form)
    assert len(form.integrals()) == 1
    assert fem.form(form) is not None


def test_constraints_select_mean_zero_modes(formulation):
    """Only MEAN_ZERO rigid-body modes are converted to functionals."""
    definition = RigidBodyConstraintDefinition(
        translation_x=RigidBodyConstraintMode.MEAN_ZERO,
        translation_y=RigidBodyConstraintMode.MEAN_ZERO,
        translation_z=RigidBodyConstraintMode.UNCONSTRAINED,
        rotation_x=RigidBodyConstraintMode.UNCONSTRAINED,
        rotation_y=RigidBodyConstraintMode.UNCONSTRAINED,
        rotation_z=RigidBodyConstraintMode.MEAN_ZERO,
    )

    constraints = formulation.constraints(definition)

    assert len(constraints) == 3

    for constraint in constraints:
        assert isinstance(constraint, ufl.Form)
        assert len(constraint.integrals()) == 1
        assert fem.form(constraint) is not None


def test_constraints_ignore_unconstrained_modes(formulation):
    """UNCONSTRAINED rigid-body modes produce no functionals."""
    definition = RigidBodyConstraintDefinition()

    constraints = formulation.constraints(definition)

    assert constraints == []


def test_constraints_ignore_zero_modes(formulation):
    """ZERO modes are not translated by the variational formulation yet."""
    definition = RigidBodyConstraintDefinition(
        translation_x=RigidBodyConstraintMode.ZERO,
        translation_y=RigidBodyConstraintMode.ZERO,
        translation_z=RigidBodyConstraintMode.ZERO,
        rotation_x=RigidBodyConstraintMode.ZERO,
        rotation_y=RigidBodyConstraintMode.ZERO,
        rotation_z=RigidBodyConstraintMode.ZERO,
    )

    constraints = formulation.constraints(definition)

    assert constraints == []


def test_translation_x_uses_x_displacement_component(formulation):
    """Translation-x integrates the x displacement component."""
    form = formulation.translation_x()
    integrand = form.integrals()[0].integrand()

    assert str(integrand) == "v_1[0]"


def test_translation_y_uses_y_displacement_component(formulation):
    """Translation-y integrates the y displacement component."""
    form = formulation.translation_y()
    integrand = form.integrals()[0].integrand()

    assert str(integrand) == "v_1[1]"


def test_translation_z_uses_z_displacement_component(formulation):
    """Translation-z integrates the z displacement component."""
    form = formulation.translation_z()
    integrand = form.integrals()[0].integrand()

    assert str(integrand) == "v_1[2]"


def test_rotation_z_has_expected_coordinate_structure(formulation):
    """Rotation-z contains the x*u_y-y*u_x rigid-body functional."""
    form = formulation.rotation_z()
    integrand = form.integrals()[0].integrand()

    assert str(integrand) == "x[0] * v_1[1] + -1 * v_1[0] * x[1]"

