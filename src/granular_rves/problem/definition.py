from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
class LoadingDefinition:
    """Definition of a quasi-static loading path.

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
    The loading definition specifies the loading path but does not apply
    the corresponding boundary condition. Application of the loading is
    handled by the simulation runner and boundary-condition layer.
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

    The definition itself does not perform any numerical operations.
    """

    name: str
    analysis: AnalysisDefinition
    geometry: Any
    mesh: MeshDefinition
    loading: LoadingDefinition
    output: OutputDefinition
