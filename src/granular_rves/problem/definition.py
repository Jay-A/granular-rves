"""Declarative definitions of simulation problems.

This module contains the data structures used to represent a complete
granular-rves problem definition independently of the numerical backend.

The definitions in this module describe what problem the user has specified.
They do not perform meshing, finite-element assembly, constraint enforcement,
or solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from granular_rves.mechanics.definitions import MechanicsDefinition
from granular_rves.mechanics.rigid_body import (
    ReferenceFace,
    RigidBodyConstraintMode,
)


@dataclass(frozen=True)
class AnalysisDefinition:
    """Definition of the simulation analysis.

    Parameters
    ----------
    type
        Analysis regime requested by the problem definition. The value is
        interpreted by the numerical orchestration layer.
    """

    type: str


@dataclass(frozen=True)
class MeshDefinition:
    """Definition of the finite-element mesh.

    Parameters
    ----------
    size
        Target characteristic mesh size used during mesh generation.
    """

    size: float


@dataclass(frozen=True)
class DirichletBoundaryDefinition:
    """Definition of a prescribed-displacement boundary constraint.

    Parameters
    ----------
    region
        Name of the physical boundary region to which the constraint applies.
    component
        Displacement component being constrained. Expected values are
        ``"x"``, ``"y"``, or ``"z"``.
    value
        Prescribed displacement value. ``None`` indicates that the value is
        supplied by the loading definition during execution.
    """

    region: str
    component: str
    value: float | None


@dataclass(frozen=True)
class BoundaryDefinition:
    """Definition of boundary constraints for a simulation problem.

    Parameters
    ----------
    dirichlet
        Mapping from boundary-region names to prescribed-displacement
        definitions.
    """

    dirichlet: dict[str, DirichletBoundaryDefinition]


@dataclass(frozen=True)
class RigidBodyConstraintDefinition:
    """Definition of rigid-body constraint modes.

    Parameters
    ----------
    reference_face
        Coordinate-aligned reference face on which global rigid-body
        reference functionals are evaluated. This does not impose a
        physical displacement constraint on the face.
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

    reference_face: ReferenceFace | None = None

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
class LoadingDefinition:
    """Definition of a loading path for a simulation.

    Parameters
    ----------
    type
        Type of loading applied to the problem.
    region
        Name of the physical region to which the loading applies.
    component
        Displacement component being loaded. Expected values are
        ``"x"``, ``"y"``, or ``"z"``.
    start
        Initial value of the loading parameter.
    end
        Final value of the loading parameter.
    steps
        Number of loading steps.
    """

    type: str
    region: str
    component: str
    start: float
    end: float
    steps: int


@dataclass(frozen=True)
class OutputDefinition:
    """Definition of simulation output.

    Parameters
    ----------
    directory
        Directory in which simulation output is written.
    """

    directory: str


@dataclass(frozen=True)
class ProblemDefinition:
    """Complete declarative definition of a granular-rves problem.

    Parameters
    ----------
    name
        Name identifying the problem.
    analysis
        Analysis regime for the simulation.
    geometry
        Declarative geometry definition.
    mesh
        Finite-element mesh definition.
    mechanics
        Mechanics model definitions.
    boundary
        Physical boundary constraints.
    loading
        Loading definition for the problem.
    output
        Output configuration.
    constraints
        Optional reference constraints on rigid-body modes. If omitted,
        all rigid-body modes are unconstrained. The reference face identifies
        the coordinate-aligned face on which global rigid-body reference
        functionals are evaluated; it does not impose a physical displacement
        constraint on that face.
    """

    name: str
    analysis: AnalysisDefinition
    geometry: Any
    mesh: MeshDefinition
    mechanics: MechanicsDefinition
    boundary: BoundaryDefinition
    loading: LoadingDefinition
    output: OutputDefinition
    constraints: RigidBodyConstraintDefinition = field(
        default_factory=RigidBodyConstraintDefinition
    )

