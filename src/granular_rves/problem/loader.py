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

from granular_rves.mechanics.rigid_body import RigidBodyConstraintMode
from granular_rves.problem.geometry.definition import GeometryDefinition
from granular_rves.problem.definition import (
    BalanceDefinition,
    BoundaryConditionDefinition,
    BoundaryDefinition,
    ConstitutiveDefinition,
    DirichletBoundaryDefinition,
    KinematicsDefinition,
    LoadingDefinition,
    MechanicsDefinition,
    MeshDefinition,
    OutputDefinition,
    ProblemDefinition,
    RigidBodyConstraintDefinition,
)


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
    The loader constructs declarative problem-definition objects only.
    Geometry construction, mesh generation, numerical assembly, and
    simulation execution are handled by downstream components.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Problem definition must be a YAML mapping.")

    return _build_problem_definition(data)


def _build_problem_definition(
    data: dict[str, Any],
) -> ProblemDefinition:
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
        If a required section is missing, contains an unsupported
        configuration, or contains a reference to an undeclared
        problem boundary.
    """
    required_sections = (
        "name",
        "analysis",
        "geometry",
        "mesh",
        "mechanics",
        "boundary",
        "loading",
        "output",
    )

    for section in required_sections:
        if section not in data:
            raise ValueError(
                f"Missing required problem section: {section!r}."
            )

    analysis_data = data["analysis"]
    geometry_data = data["geometry"]
    mesh_data = data["mesh"]
    mechanics_data = data["mechanics"]
    boundary_data = data["boundary"]
    loading_data = data["loading"]
    output_data = data["output"]

    if not isinstance(analysis_data, dict):
        raise ValueError("analysis must be a YAML mapping.")

    if not isinstance(mesh_data, dict):
        raise ValueError("mesh must be a YAML mapping.")

    if not isinstance(mechanics_data, dict):
        raise ValueError("mechanics must be a YAML mapping.")

    geometry = _build_geometry(geometry_data)
    mechanics = _build_mechanics(mechanics_data)
    boundary = _build_boundary(boundary_data)
    constraints = _build_constraints(data.get("constraints"))
    loading = _build_loading(loading_data)

    _validate_boundary_references(
        boundary=boundary,
        constraints=constraints,
        loading=loading,
    )

    if not isinstance(output_data, dict):
        raise ValueError("output must be a YAML mapping.")

    try:
        output_directory = output_data["directory"]
    except KeyError as exc:
        raise ValueError(
            f"Missing required output field: {exc.args[0]!r}."
        ) from exc

    try:
        analysis_type = analysis_data["type"]
    except KeyError as exc:
        raise ValueError(
            f"Missing required analysis field: {exc.args[0]!r}."
        ) from exc

    try:
        mesh_size = mesh_data["size"]
    except KeyError as exc:
        raise ValueError(
            f"Missing required mesh field: {exc.args[0]!r}."
        ) from exc

    return ProblemDefinition(
        name=str(data["name"]),
        analysis=str(analysis_type),
        geometry=geometry,
        mesh=MeshDefinition(
            size=float(mesh_size),
            order=int(mesh_data.get("order", 1)),
        ),
        mechanics=mechanics,
        boundary=boundary,
        constraints=constraints,
        loading=loading,
        output=OutputDefinition(
            directory=str(output_directory),
        ),
    )


def _validate_boundary_references(
    boundary: BoundaryConditionDefinition,
    constraints: RigidBodyConstraintDefinition,
    loading: LoadingDefinition,
) -> None:
    """Validate problem-level references to declared boundaries.

    Parameters
    ----------
    boundary
        Problem boundary registry.
    constraints
        Rigid-body constraint definition.
    loading
        Loading definition.

    Raises
    ------
    ValueError
        If ``constraints.reference_boundary`` or ``loading.boundary``
        refers to a problem boundary that is not declared.
    """
    declared_boundaries = boundary.boundaries

    reference_boundary = constraints.reference_boundary

    if (
        reference_boundary is not None
        and reference_boundary not in declared_boundaries
    ):
        raise ValueError(
            "constraints.reference_boundary "
            f"{reference_boundary!r} is not declared "
            "in the problem boundary registry."
        )

    if loading.boundary not in declared_boundaries:
        raise ValueError(
            f"loading.boundary {loading.boundary!r} is not declared "
            "in the problem boundary registry."
        )


def _build_constraints(
    data: dict[str, Any] | None,
) -> RigidBodyConstraintDefinition:
    """Construct the configured rigid-body constraint definition.

    Parameters
    ----------
    data
        Parsed ``constraints`` section from the YAML problem definition.
        If ``None``, all rigid-body modes remain unconstrained.

    Returns
    -------
    RigidBodyConstraintDefinition
        Structured rigid-body constraint configuration.

    Raises
    ------
    ValueError
        If the constraints configuration is malformed or contains an
        unsupported rigid-body constraint mode.

    Notes
    -----
    ``reference_boundary`` identifies a problem boundary on which the
    global rigid-body reference functionals are evaluated. It does not
    directly identify a geometry face or mesh entity.

    Boundary-name validation is performed by
    :func:`_validate_boundary_references`, because it requires the
    complete problem boundary registry.
    """
    if data is None:
        return RigidBodyConstraintDefinition()

    if not isinstance(data, dict):
        raise ValueError("constraints must be a YAML mapping.")

    reference_boundary = data.get("reference_boundary")

    if reference_boundary is not None and not isinstance(
        reference_boundary,
        str,
    ):
        raise ValueError(
            "constraints.reference_boundary must be a string."
        )

    rigid_body_data = data.get("rigid_body", {})

    if not isinstance(rigid_body_data, dict):
        raise ValueError(
            "constraints.rigid_body must be a YAML mapping."
        )

    translation_data = rigid_body_data.get("translation", {})
    rotation_data = rigid_body_data.get("rotation", {})

    if not isinstance(translation_data, dict):
        raise ValueError(
            "constraints.rigid_body.translation must be a YAML mapping."
        )

    if not isinstance(rotation_data, dict):
        raise ValueError(
            "constraints.rigid_body.rotation must be a YAML mapping."
        )

    return RigidBodyConstraintDefinition(
        reference_boundary=reference_boundary,
        translation_x=_parse_constraint_mode(
            translation_data.get(
                "x",
                RigidBodyConstraintMode.UNCONSTRAINED.value,
            ),
            "constraints.rigid_body.translation.x",
        ),
        translation_y=_parse_constraint_mode(
            translation_data.get(
                "y",
                RigidBodyConstraintMode.UNCONSTRAINED.value,
            ),
            "constraints.rigid_body.translation.y",
        ),
        translation_z=_parse_constraint_mode(
            translation_data.get(
                "z",
                RigidBodyConstraintMode.UNCONSTRAINED.value,
            ),
            "constraints.rigid_body.translation.z",
        ),
        rotation_x=_parse_constraint_mode(
            rotation_data.get(
                "x",
                RigidBodyConstraintMode.UNCONSTRAINED.value,
            ),
            "constraints.rigid_body.rotation.x",
        ),
        rotation_y=_parse_constraint_mode(
            rotation_data.get(
                "y",
                RigidBodyConstraintMode.UNCONSTRAINED.value,
            ),
            "constraints.rigid_body.rotation.y",
        ),
        rotation_z=_parse_constraint_mode(
            rotation_data.get(
                "z",
                RigidBodyConstraintMode.UNCONSTRAINED.value,
            ),
            "constraints.rigid_body.rotation.z",
        ),
    )


def _parse_constraint_mode(
    value: Any,
    field_name: str,
) -> RigidBodyConstraintMode:
    """Parse a rigid-body constraint mode.

    Parameters
    ----------
    value
        YAML value containing the constraint mode.
    field_name
        Configuration field name used in validation messages.

    Returns
    -------
    RigidBodyConstraintMode
        Parsed constraint mode.

    Raises
    ------
    ValueError
        If the value is not a supported constraint mode.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string."
        )

    try:
        return RigidBodyConstraintMode(value)
    except ValueError as exc:
        allowed = tuple(
            mode.value for mode in RigidBodyConstraintMode
        )

        raise ValueError(
            f"Unsupported rigid-body constraint mode {value!r} "
            f"for {field_name}. Expected one of {allowed}."
        ) from exc


def _build_boundary(
    data: dict[str, Any],
) -> BoundaryConditionDefinition:
    """Construct the configured problem boundaries.

    Parameters
    ----------
    data
        Parsed ``boundary`` section from the YAML problem definition.

    Returns
    -------
    BoundaryConditionDefinition
        Structured problem boundaries and boundary conditions.

    Raises
    ------
    ValueError
        If the boundary definition is malformed, refers to an
        undeclared problem boundary, or contains an unsupported
        boundary-condition definition.
    """
    if not isinstance(data, dict):
        raise ValueError("boundary must be a YAML mapping.")

    dirichlet_data = data.get("dirichlet", {})

    if not isinstance(dirichlet_data, dict):
        raise ValueError(
            "boundary.dirichlet must be a YAML mapping."
        )

    boundaries: dict[str, BoundaryDefinition] = {}

    for boundary_name, definition_data in data.items():
        if boundary_name == "dirichlet":
            continue

        if not isinstance(definition_data, dict):
            raise ValueError(
                f"Problem boundary {boundary_name!r} must be "
                "a YAML mapping."
            )

        try:
            face = definition_data["face"]
        except KeyError as exc:
            raise ValueError(
                f"Missing required boundary field {exc.args[0]!r} "
                f"for problem boundary {boundary_name!r}."
            ) from exc

        if not isinstance(face, str):
            raise ValueError(
                "Boundary field 'face' for problem boundary "
                f"{boundary_name!r} must be a string."
            )

        boundaries[str(boundary_name)] = BoundaryDefinition(
            face=face,
        )

    dirichlet: dict[str, DirichletBoundaryDefinition] = {}

    for boundary_name, definition_data in dirichlet_data.items():
        boundary_name = str(boundary_name)

        if boundary_name not in boundaries:
            raise ValueError(
                f"Dirichlet boundary {boundary_name!r} is not "
                "declared in the problem boundary registry."
            )

        if not isinstance(definition_data, dict):
            raise ValueError(
                "Dirichlet boundary definition for "
                f"{boundary_name!r} must be a YAML mapping."
            )

        try:
            component = definition_data["component"]
        except KeyError as exc:
            raise ValueError(
                f"Missing required Dirichlet field {exc.args[0]!r} "
                f"for boundary {boundary_name!r}."
            ) from exc

        if not isinstance(component, str):
            raise ValueError(
                f"Dirichlet component for boundary "
                f"{boundary_name!r} must be a string."
            )

        value = definition_data.get("value")

        if value is not None:
            value = float(value)

        dirichlet[boundary_name] = DirichletBoundaryDefinition(
            component=component,
            value=value,
        )

    return BoundaryConditionDefinition(
        boundaries=boundaries,
        dirichlet=dirichlet,
    )


def _build_loading(
    data: dict[str, Any],
) -> LoadingDefinition:
    """Construct the configured loading definition.

    Parameters
    ----------
    data
        Parsed ``loading`` section from the YAML problem definition.

    Returns
    -------
    LoadingDefinition
        Structured loading configuration.

    Raises
    ------
    ValueError
        If the loading configuration is malformed or required fields
        are missing.

    Notes
    -----
    Validation that ``boundary`` refers to a declared problem boundary
    is performed by :func:`_validate_boundary_references`, because that
    requires the complete problem boundary registry.
    """
    if not isinstance(data, dict):
        raise ValueError("loading must be a YAML mapping.")

    try:
        loading_type = data["type"]
        boundary = data["boundary"]
        component = data["component"]
        value = data["value"]
    except KeyError as exc:
        raise ValueError(
            f"Missing required loading field: {exc.args[0]!r}."
        ) from exc

    if not isinstance(boundary, str):
        raise ValueError("loading.boundary must be a string.")

    if not isinstance(component, str):
        raise ValueError("loading.component must be a string.")

    return LoadingDefinition(
        type=str(loading_type),
        boundary=boundary,
        component=component,
        value=float(value),
    )


def _build_mechanics(
    data: dict[str, Any],
) -> MechanicsDefinition:
    """Construct the configured mechanics definition.

    Parameters
    ----------
    data
        Parsed mechanics configuration.

    Returns
    -------
    MechanicsDefinition
        Structured mechanics definition.

    Raises
    ------
    ValueError
        If a mechanics section is malformed or does not contain
        exactly one configured model.
    """
    if not isinstance(data, dict):
        raise ValueError("mechanics must be a YAML mapping.")

    try:
        kinematics_data = data["kinematics"]
        constitutive_data = data["constitutive"]
        balance_data = data["balance"]
    except KeyError as exc:
        raise ValueError(
            f"Missing required mechanics section: {exc.args[0]!r}."
        ) from exc

    kinematics_type, kinematics_parameters = _single_model(
        kinematics_data,
        "mechanics.kinematics",
    )

    constitutive_type, constitutive_parameters = _single_model(
        constitutive_data,
        "mechanics.constitutive",
    )

    balance_type, balance_parameters = _single_model(
        balance_data,
        "mechanics.balance",
    )

    return MechanicsDefinition(
        kinematics=KinematicsDefinition(
            models={
                kinematics_type: kinematics_parameters,
            },
        ),
        constitutive=ConstitutiveDefinition(
            models={
                constitutive_type: constitutive_parameters,
            },
        ),
        balance=BalanceDefinition(
            models={
                balance_type: balance_parameters,
            },
        ),
    )


def _single_model(
    data: Any,
    field_name: str,
) -> tuple[str, dict[str, Any]]:
    """Extract the single configured model from a mechanics section.

    Parameters
    ----------
    data
        Parsed YAML mechanics model mapping.
    field_name
        Configuration field name used in validation messages.

    Returns
    -------
    tuple[str, dict[str, Any]]
        Model name and its parameter mapping.

    Raises
    ------
    ValueError
        If the section is not a mapping or does not contain exactly
        one model.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"{field_name} must be a YAML mapping."
        )

    if len(data) != 1:
        raise ValueError(
            f"{field_name} must contain exactly one configured model."
        )

    model_name, parameters = next(iter(data.items()))

    if parameters is None:
        parameters = {}

    if not isinstance(parameters, dict):
        raise ValueError(
            f"Configuration for {field_name}.{model_name} "
            "must be a YAML mapping."
        )

    return str(model_name), dict(parameters)


def _build_geometry(
    data: dict[str, Any],
) -> GeometryDefinition:
    """Construct the declarative geometry definition.

    Parameters
    ----------
    data
        Parsed geometry configuration.

    Returns
    -------
    GeometryDefinition
        Declarative geometry configuration.

    Raises
    ------
    ValueError
        If the geometry configuration is malformed or unsupported.
    """
    if not isinstance(data, dict):
        raise ValueError("geometry must be a YAML mapping.")

    try:
        geometry_type = data["type"]
    except KeyError as exc:
        raise ValueError(
            f"Missing required geometry field: {exc.args[0]!r}."
        ) from exc

    if not isinstance(geometry_type, str):
        raise ValueError("geometry.type must be a string.")

    name = data.get("name")

    if name is not None and not isinstance(name, str):
        raise ValueError("geometry.name must be a string.")

    parameters = {
        key: value
        for key, value in data.items()
        if key not in {"name", "type"}
    }

    return GeometryDefinition(
        type=geometry_type,
        parameters=parameters,
        name=name,
    )


