"""Tests for loading simulation problem definitions from YAML files."""

from pathlib import Path

import pytest
import yaml

from granular_rves.mechanics.rigid_body import RigidBodyConstraintMode
from granular_rves.problem.definition import (
    BalanceDefinition,
    ConstitutiveDefinition,
    KinematicsDefinition,
    MechanicsDefinition,
)
from granular_rves.problem.loader import load_problem


def _write_problem(path: Path) -> None:
    """Write a minimal valid problem definition to a YAML file."""
    data = {
        "name": "example",
        "analysis": {
            "type": "steady",
        },
        "geometry": {
            "type": "cylinder",
            "x0": [0.0, 0.0, 0.0],
            "x1": [0.0, 0.0, 1.0],
            "radius": 1.0,
        },
        "mesh": {
            "size": 0.25,
        },
        "mechanics": {
            "kinematics": {
                "small_strain": {},
            },
            "constitutive": {
                "linear_elastic": {
                    "youngs_modulus": 1.0e6,
                    "poisson_ratio": 0.3,
                },
            },
            "balance": {
                "momentum": {},
            },
        },
        "boundary": {
            "bottom": {
                "face": "bottom",
            },
            "top": {
                "face": "top",
            },
            "lateral": {
                "face": "lateral",
            },
            "dirichlet": {
                "bottom": {
                    "component": "z",
                    "value": 0.0,
                },
                "top": {
                    "component": "z",
                },
            },
        },
        "constraints": {
            "reference_boundary": "bottom",
        },
        "loading": {
            "type": "displacement",
            "boundary": "top",
            "component": "z",
            "value": -0.01,
        },
        "output": {
            "directory": "output/example",
            "reaction_history": True,
            "reactions": [
                {
                    "name": "top_force",
                    "boundary": "top",
                    "component": "z",
                },
                {
                    "name": "bottom_force",
                    "boundary": "bottom",
                    "component": "z",
                },
            ],
        },
    }

    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )


def test_load_problem(tmp_path: Path) -> None:
    """Load a valid YAML problem definition."""
    path = tmp_path / "problem.yaml"
    _write_problem(path)

    problem = load_problem(path)

    assert problem.name == "example"
    assert problem.analysis == "steady"

    assert problem.geometry.type == "cylinder"
    assert problem.geometry.name == "cylinder"
    assert problem.geometry.parameters == {
        "x0": [0.0, 0.0, 0.0],
        "x1": [0.0, 0.0, 1.0],
        "radius": 1.0,
    }

    assert problem.mesh.size == 0.25
    assert problem.mesh.order == 1

    assert isinstance(problem.mechanics, MechanicsDefinition)

    assert isinstance(
        problem.mechanics.kinematics,
        KinematicsDefinition,
    )
    assert problem.mechanics.kinematics.models == {
        "small_strain": {},
    }

    assert isinstance(
        problem.mechanics.constitutive,
        ConstitutiveDefinition,
    )
    assert problem.mechanics.constitutive.models == {
        "linear_elastic": {
            "youngs_modulus": 1.0e6,
            "poisson_ratio": 0.3,
        },
    }

    assert isinstance(
        problem.mechanics.balance,
        BalanceDefinition,
    )
    assert problem.mechanics.balance.models == {
        "momentum": {},
    }

    assert problem.boundary.boundaries["bottom"].face == "bottom"
    assert problem.boundary.boundaries["top"].face == "top"
    assert problem.boundary.boundaries["lateral"].face == "lateral"

    assert problem.boundary.dirichlet["bottom"].component == "z"
    assert problem.boundary.dirichlet["bottom"].value == 0.0

    assert problem.boundary.dirichlet["top"].component == "z"
    assert problem.boundary.dirichlet["top"].value is None

    assert problem.constraints.reference_boundary == "bottom"
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

    assert problem.loading.type == "displacement"
    assert problem.loading.boundary == "top"
    assert problem.loading.component == "z"
    assert problem.loading.value == -0.01

    assert problem.output.directory == "output/example"
    assert problem.output.reaction_history is True
    assert len(problem.output.reactions) == 2

    assert problem.output.reactions[0].name == "top_force"
    assert problem.output.reactions[0].boundary == "top"
    assert problem.output.reactions[0].component == "z"

    assert problem.output.reactions[1].name == "bottom_force"
    assert problem.output.reactions[1].boundary == "bottom"
    assert problem.output.reactions[1].component == "z"


def test_load_problem_accepts_string_path(tmp_path: Path) -> None:
    """Load a problem definition when the path is supplied as a string."""
    path = tmp_path / "problem.yaml"
    _write_problem(path)

    problem = load_problem(str(path))

    assert problem.name == "example"


def test_load_problem_missing_section(tmp_path: Path) -> None:
    """Reject a YAML problem definition with a missing required section."""
    path = tmp_path / "problem.yaml"
    _write_problem(path)

    data = yaml.safe_load(
        path.read_text(encoding="utf-8"),
    )
    del data["mechanics"]

    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Missing required problem section: 'mechanics'",
    ):
        load_problem(path)


def test_load_problem_rejects_non_mapping(tmp_path: Path) -> None:
    """Reject a YAML document whose root is not a mapping."""
    path = tmp_path / "problem.yaml"

    path.write_text(
        "- invalid\n- problem\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Problem definition must be a YAML mapping",
    ):
        load_problem(path)


def test_load_problem_rigid_body_constraints(
    tmp_path: Path,
) -> None:
    """Load explicitly configured rigid-body constraint modes."""
    path = tmp_path / "problem.yaml"
    _write_problem(path)

    data = yaml.safe_load(
        path.read_text(encoding="utf-8"),
    )

    data["constraints"] = {
        "reference_boundary": "bottom",
        "rigid_body": {
            "translation": {
                "x": "mean_zero",
                "y": "mean_zero",
                "z": "unconstrained",
            },
            "rotation": {
                "x": "unconstrained",
                "y": "unconstrained",
                "z": "zero",
            },
        },
    }

    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    problem = load_problem(path)

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
        is RigidBodyConstraintMode.ZERO
    )


def test_load_problem_rejects_invalid_rigid_body_constraint(
    tmp_path: Path,
) -> None:
    """Reject an unsupported rigid-body constraint mode."""
    path = tmp_path / "problem.yaml"
    _write_problem(path)

    data = yaml.safe_load(
        path.read_text(encoding="utf-8"),
    )

    data["constraints"] = {
        "reference_boundary": "bottom",
        "rigid_body": {
            "translation": {
                "x": "invalid",
            },
        },
    }

    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported rigid-body constraint mode",
    ):
        load_problem(path)


def test_load_problem_rejects_undeclared_dirichlet_boundary(
    tmp_path: Path,
) -> None:
    """Reject a Dirichlet condition for an unknown problem boundary."""
    path = tmp_path / "problem.yaml"
    _write_problem(path)

    data = yaml.safe_load(
        path.read_text(encoding="utf-8"),
    )

    data["boundary"]["dirichlet"]["unknown"] = {
        "component": "x",
        "value": 0.0,
    }

    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Dirichlet boundary 'unknown' is not declared",
    ):
        load_problem(path)


def test_load_problem_preserves_boundary_registry(
    tmp_path: Path,
) -> None:
    """Load generic problem boundaries independently of conditions."""
    path = tmp_path / "problem.yaml"
    _write_problem(path)

    problem = load_problem(path)

    assert set(problem.boundary.boundaries) == {
        "bottom",
        "top",
        "lateral",
    }

    assert problem.boundary.boundaries["bottom"].face == "bottom"
    assert problem.boundary.boundaries["top"].face == "top"
    assert problem.boundary.boundaries["lateral"].face == "lateral"


def test_load_problem_rejects_undeclared_reference_boundary(
    tmp_path: Path,
) -> None:
    """Reject a rigid-body reference boundary that is not registered."""
    path = tmp_path / "problem.yaml"
    _write_problem(path)

    data = yaml.safe_load(
        path.read_text(encoding="utf-8"),
    )

    data["constraints"]["reference_boundary"] = "unknown"

    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    # The loader currently validates the reference-boundary name only
    # structurally; boundary-registry validation belongs in the loader's
    # problem-level consistency checks.
    with pytest.raises(
        ValueError,
        match="reference_boundary",
    ):
        load_problem(path)


def test_load_problem_rejects_undeclared_loading_boundary(
    tmp_path: Path,
) -> None:
    """Reject a loading boundary that is not registered."""
    path = tmp_path / "problem.yaml"
    _write_problem(path)

    data = yaml.safe_load(
        path.read_text(encoding="utf-8"),
    )

    data["loading"]["boundary"] = "unknown"

    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    # The loader currently validates the loading boundary name only
    # structurally; boundary-registry validation belongs in the loader's
    # problem-level consistency checks.
    with pytest.raises(
        ValueError,
        match="loading.boundary",
    ):
        load_problem(path)


def test_load_problem_defaults_geometry_name_to_type(
    tmp_path: Path,
) -> None:
    """Default the geometry name to its geometry type."""
    path = tmp_path / "problem.yaml"
    _write_problem(path)

    problem = load_problem(path)

    assert problem.geometry.name == "cylinder"


def test_load_problem_preserves_geometry_name(
    tmp_path: Path,
) -> None:
    """Preserve an explicitly configured geometry name."""
    path = tmp_path / "problem.yaml"
    _write_problem(path)

    data = yaml.safe_load(
        path.read_text(encoding="utf-8"),
    )
    data["geometry"]["name"] = "specimen"

    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    problem = load_problem(path)

    assert problem.geometry.name == "specimen"
    assert problem.geometry.parameters == {
        "x0": [0.0, 0.0, 0.0],
        "x1": [0.0, 0.0, 1.0],
        "radius": 1.0,
    }
