"""Declarative definitions of simulation problems.

This module contains the problem-level data structures used to describe
a simulation independently of numerical discretization and solver
implementation.

The definitions in this module describe what problem is being solved.
They do not construct meshes, UFL forms, boundary measures, PETSc
objects, or numerical solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from granular_rves.problem.geometry.definition import GeometryDefinition
from granular_rves.mechanics.rigid_body import RigidBodyConstraintMode


@dataclass(frozen=True)
class MeshDefinition:
    """Definition of the mesh settings.

    Parameters
    ----------
    size
        Characteristic mesh size.
    order
        Polynomial order of the finite-element approximation.
    """

    size: float
    order: int = 1


@dataclass(frozen=True)
class KinematicsDefinition:
    """Definition of the kinematic model.

    Parameters
    ----------
    models
        Mapping of kinematic model names to their configuration.
    """

    models: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ConstitutiveDefinition:
    """Definition of the constitutive model.

    Parameters
    ----------
    models
        Mapping of constitutive model names to their configuration.
    """

    models: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class BalanceDefinition:
    """Definition of the balance laws.

    Parameters
    ----------
    models
        Mapping of balance-law names to their configuration.
    """

    models: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class MechanicsDefinition:
    """Definition of the mechanics models.

    Parameters
    ----------
    kinematics
        Kinematic model definitions.
    constitutive
        Constitutive model definitions.
    balance
        Balance-law definitions.
    """

    kinematics: KinematicsDefinition
    constitutive: ConstitutiveDefinition
    balance: BalanceDefinition


@dataclass(frozen=True)
class BoundaryDefinition:
    """Definition of a problem-level boundary.

    A problem boundary associates a named physical boundary with a
    geometry face. The association is declarative and does not resolve
    the geometry face to mesh entities.

    Parameters
    ----------
    face
        Name of the geometry face associated with this problem boundary.
    """

    face: str


@dataclass(frozen=True)
class DirichletBoundaryDefinition:
    """Definition of a Dirichlet condition on a problem boundary.

    Parameters
    ----------
    component
        Displacement component constrained by the Dirichlet condition.
    value
        Prescribed value, when applicable. A value of ``None`` indicates
        that the current value is supplied separately by the loading
        configuration.
    """

    component: str
    value: float | None = None


@dataclass(frozen=True)
class BoundaryConditionDefinition:
    """Collection of problem boundaries and boundary conditions.

    Parameters
    ----------
    boundaries
        Mapping from problem boundary names to their geometry-face
        associations.
    dirichlet
        Mapping from problem boundary names to Dirichlet conditions.
    """

    boundaries: Mapping[str, BoundaryDefinition] = field(
        default_factory=dict
    )
    dirichlet: Mapping[str, DirichletBoundaryDefinition] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class LoadingDefinition:
    """Definition of the applied loading.

    Parameters
    ----------
    type
        Loading type identifier.
    boundary
        Name of the problem boundary on which the loading is applied.
    component
        Component affected by the loading.
    value
        Applied loading value.
    """

    type: str
    boundary: str
    component: str
    value: float


@dataclass(frozen=True)
class OutputDefinition:
    """Definition of simulation output.

    Parameters
    ----------
    directory
        Output directory.
    """

    directory: str


@dataclass(frozen=True)
class RigidBodyConstraintDefinition:
    """Definition of rigid-body constraint modes.

    Parameters
    ----------
    reference_boundary
        Name of the problem boundary used as the global reference
        region for rigid-body constraint functionals. This refers to a
        problem boundary, not directly to a geometry face.
    translation_x
        Constraint mode for rigid translation along the global x axis.
    translation_y
        Constraint mode for rigid translation along the global y axis.
    translation_z
        Constraint mode for rigid translation along the global z axis.
    rotation_x
        Constraint mode for rigid rotation about the global x axis.
    rotation_y
        Constraint mode for rigid rotation about the global y axis.
    rotation_z
        Constraint mode for rigid rotation about the global z axis.

    Notes
    -----
    All rigid-body modes default to
    :attr:`RigidBodyConstraintMode.UNCONSTRAINED`. This allows problem
    definitions that are already physically constrained to omit a
    ``constraints`` section entirely.
    """

    reference_boundary: str | None = None
    translation_x: RigidBodyConstraintMode = (
        RigidBodyConstraintMode.UNCONSTRAINED
    )
    translation_y: RigidBodyConstraintMode = (
        RigidBodyConstraintMode.UNCONSTRAINED
    )
    translation_z: RigidBodyConstraintMode = (
        RigidBodyConstraintMode.UNCONSTRAINED
    )
    rotation_x: RigidBodyConstraintMode = (
        RigidBodyConstraintMode.UNCONSTRAINED
    )
    rotation_y: RigidBodyConstraintMode = (
        RigidBodyConstraintMode.UNCONSTRAINED
    )
    rotation_z: RigidBodyConstraintMode = (
        RigidBodyConstraintMode.UNCONSTRAINED
    )


@dataclass(frozen=True)
class ProblemDefinition:
    """Complete declarative definition of a simulation problem.

    Parameters
    ----------
    name
        Problem name.
    analysis
        Analysis type identifier.
    geometry
        Geometry definition.
    mesh
        Mesh definition.
    mechanics
        Mechanics model definitions.
    boundary
        Problem boundary and boundary-condition definitions.
    constraints
        Rigid-body constraint definitions.
    loading
        Loading definition.
    output
        Output definition.
    """

    name: str
    analysis: str
    geometry: GeometryDefinition
    mesh: MeshDefinition
    mechanics: MechanicsDefinition
    boundary: BoundaryConditionDefinition
    constraints: RigidBodyConstraintDefinition = field(
        default_factory=RigidBodyConstraintDefinition
    )
    loading: LoadingDefinition | None = None
    output: OutputDefinition | None = None


