from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from granular_rves.mechanics.definitions import MechanicsDefinition


@dataclass(frozen=True)
class AnalysisDefinition:
    """Definition of the simulation analysis.

    Parameters
    ----------
    type
        Analysis regime. Supported values are ``"static"``,
        ``"quasi_static"``, and ``"dynamic"``.

    Notes
    -----
    The analysis definition describes the physical analysis regime rather
    than the numerical solver used to execute it.
    """

    type: str


@dataclass(frozen=True)
class MeshDefinition:
    """Definition of the finite-element mesh.

    Parameters
    ----------
    size
        Target characteristic mesh size passed to the mesh-generation
        procedure.

    Notes
    -----
    The mesh definition describes the requested discretization. Mesh
    generation and conversion to a DOLFINx mesh are handled by the
    meshing layer.
    """

    size: float


@dataclass(frozen=True)
class DirichletBoundaryDefinition:
    """Definition of a prescribed-displacement boundary constraint.

    Parameters
    ----------
    region
        Name of the geometric boundary region on which the constraint
        is applied.
    component
        Spatial displacement component constrained by the boundary
        condition, such as ``"x"``, ``"y"``, or ``"z"``.
    value
        Prescribed displacement value. ``None`` indicates that the
        current value is supplied by a loading definition during
        numerical execution.

    Notes
    -----
    This definition describes the physical constraint independently of
    the numerical representation used to enforce it. In particular, it
    does not contain DOLFINx facet indices, degrees of freedom, function
    spaces, or boundary-condition objects.

    For a loading-controlled boundary, ``value`` is ``None`` and the
    corresponding current value is supplied by the numerical loading
    procedure at each applicable loading step.
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
        Prescribed-displacement boundary constraints.

    Notes
    -----
    Only explicitly constrained boundaries are represented here.
    Boundaries absent from ``dirichlet`` are not implicitly constrained.

    Consequently, a geometric boundary such as a lateral surface can
    remain unconstrained and therefore receive its natural boundary
    condition from the variational formulation.
    """

    dirichlet: dict[str, DirichletBoundaryDefinition]


@dataclass(frozen=True)
class LoadingDefinition:
    """Definition of a loading path for a simulation.

    Parameters
    ----------
    type
        Loading type, such as ``"displacement"`` or ``"traction"``.
    region
        Named geometric region to which the loading is applied.
    component
        Spatial component affected by the loading, such as ``"x"``,
        ``"y"``, or ``"z"``.
    start
        Initial value of the prescribed loading quantity.
    end
        Final value of the prescribed loading quantity.
    steps
        Number of increments used to apply the loading path.

    Notes
    -----
    The loading definition describes the requested evolution of a
    loading quantity over the simulation. It does not itself apply a
    boundary condition, construct a numerical state, or perform
    numerical stepping.

    The numerical execution layer is responsible for interpreting this
    loading path according to the selected analysis and supplying the
    current loading value to the appropriate numerical mechanism.

    Boundary constraints are represented separately by
    :class:`BoundaryDefinition`.
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

    Notes
    -----
    The output definition specifies requested output configuration only.
    Writing simulation data is handled by the output layer.
    """

    directory: str


@dataclass(frozen=True)
class ProblemDefinition:
    """Complete definition of a granular-rves simulation problem.

    Parameters
    ----------
    name
        Unique or descriptive name assigned to the simulation problem.
    analysis
        Definition of the physical analysis regime.
    geometry
        Geometry object defining the physical domain.
    mesh
        Definition of the requested finite-element discretization.
    mechanics
        Definition of the mechanics models used by the simulation.
    boundary
        Definition of the prescribed boundary constraints.
    loading
        Definition of the applied loading path.
    output
        Definition of simulation output.

    Notes
    -----
    ``ProblemDefinition`` is the contract between problem configuration
    and simulation execution. A problem loader constructs this object
    from an external problem specification, while the simulation runner
    consumes it to orchestrate geometry construction, mesh generation,
    mechanics, solution, and output.

    The definition itself does not perform any numerical operations and
    does not depend on DOLFINx, UFL, PETSc, or a particular numerical
    solution procedure.
    """

    name: str
    analysis: AnalysisDefinition
    geometry: Any
    mesh: MeshDefinition
    mechanics: MechanicsDefinition
    boundary: BoundaryDefinition
    loading: LoadingDefinition
    output: OutputDefinition
