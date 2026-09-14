"""Load simulation problem definitions from YAML configuration files.

This module translates an external YAML problem specification into the
structured :class:`~granular_rves.problem.definition.ProblemDefinition`
used by the simulation framework.

The loader is responsible only for configuration parsing and construction
of problem-definition objects. It does not perform geometry construction,
mesh generation, numerical assembly, or simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from granular_rves.mechanics.definitions import (
    BalanceDefinition,
    ConstitutiveDefinition,
    KinematicsDefinition,
    MechanicsDefinition,
)
from granular_rves.problem.definition import (
    AnalysisDefinition,
    LoadingDefinition,
    MeshDefinition,
    OutputDefinition,
    ProblemDefinition,
)
from granular_rves.problem.geometry.geometry_types.cylinder import Cylinder


def load_problem(path: str | Path) -> ProblemDefinition:
    """Load a simulation problem definition from a YAML file.

    Parameters
    ----------
    path
        Path to the YAML problem-definition file.

    Returns
    -------
    ProblemDefinition
        Structured representation of the configured simulation problem.

    Raises
    ------
    FileNotFoundError
        If the specified YAML file does not exist.
    ValueError
        If the YAML document does not contain the required problem
        definition fields or contains an unsupported configuration.
    yaml.YAMLError
        If the YAML file cannot be parsed.

    Examples
    --------
    Load a problem definition from a YAML file::

        problem = load_problem("examples/example.yaml")

    A :class:`pathlib.Path` may also be supplied::

        problem = load_problem(Path("examples/example.yaml"))

    Notes
    -----
    The loader constructs configuration and geometry objects but does not
    perform any numerical operations. Geometry construction, mesh
    generation, and simulation execution are handled by downstream
    components.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Problem definition must be a YAML mapping.")

    return _build_problem_definition(data)


def _build_problem_definition(data: dict[str, Any]) -> ProblemDefinition:
    """Construct a problem definition from parsed YAML data.

    Parameters
    ----------
    data
        Parsed YAML problem definition.

    Returns
    -------
    ProblemDefinition
        Structured problem definition.

    Raises
    ------
    ValueError
        If a required section is missing or contains an unsupported
        configuration.
    """
    try:
        analysis_data = data["analysis"]
        geometry_data = data["geometry"]
        mesh_data = data["mesh"]
        mechanics_data = data["mechanics"]
        loading_data = data["loading"]
        output_data = data["output"]
    except KeyError as exc:
        raise ValueError(
            f"Missing required problem section: {exc.args[0]!r}."
        ) from exc

    geometry = _build_geometry(geometry_data)
    mechanics = _build_mechanics(mechanics_data)

    return ProblemDefinition(
        name=str(data["name"]),
        analysis=AnalysisDefinition(
            type=str(analysis_data["type"]),
        ),
        geometry=geometry,
        mesh=MeshDefinition(
            size=float(mesh_data["size"]),
        ),
        mechanics=mechanics,
        loading=LoadingDefinition(
            type=str(loading_data["type"]),
            region=str(loading_data["region"]),
            component=str(loading_data["component"]),
            start=float(loading_data["start"]),
            end=float(loading_data["end"]),
            steps=int(loading_data["steps"]),
        ),
        output=OutputDefinition(
            directory=str(output_data["directory"]),
        ),
    )


def _build_mechanics(data: dict[str, Any]) -> MechanicsDefinition:
    """Construct the configured mechanics definition.

    Parameters
    ----------
    data
        Parsed mechanics configuration.

    Returns
    -------
    MechanicsDefinition
        Structured mechanics definition.
    """
    kinematics_data = data["kinematics"]
    constitutive_data = data["constitutive"]
    balance_data = data["balance"]

    kinematics_type, kinematics_parameters = next(
        iter(kinematics_data.items())
    )
    constitutive_type, constitutive_parameters = next(
        iter(constitutive_data.items())
    )
    balance_type, balance_parameters = next(
        iter(balance_data.items())
    )

    return MechanicsDefinition(
        kinematics=KinematicsDefinition(
            type=str(kinematics_type),
            parameters=kinematics_parameters,
        ),
        constitutive=ConstitutiveDefinition(
            type=str(constitutive_type),
            parameters=constitutive_parameters,
        ),
        balance=BalanceDefinition(
            type=str(balance_type),
            parameters=balance_parameters,
        ),
    )


def _build_geometry(data: dict[str, Any]) -> Cylinder:
    """Construct the configured geometry object.

    Parameters
    ----------
    data
        Parsed geometry configuration.

    Returns
    -------
    Cylinder
        Constructed cylinder geometry.

    Raises
    ------
    ValueError
        If the geometry type is unsupported.

    Notes
    -----
    Cylinder geometry is the only supported geometry type in the initial
    problem-definition loader. Additional geometry types can be added
    without changing the public :func:`load_problem` interface.
    """
    geometry_type = data["type"]

    if geometry_type == "cylinder":
        return Cylinder(
            x0=data["x0"],
            x1=data["x1"],
            radius=data["radius"],
        )

    raise ValueError(
        f"Unsupported geometry type: {geometry_type!r}."
    )


