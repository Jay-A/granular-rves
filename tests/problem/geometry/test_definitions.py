import pytest

from granular_rves.problem.geometry.definitions import GeometryDefinition


def test_valid_geometry_definition() -> None:
    definition = GeometryDefinition(
        name="specimen",
        type="cylinder",
        parameters={
            "x0": [0.0, 0.0, 0.0],
            "x1": [0.0, 0.0, 0.01],
            "radius": 0.005,
        },
    )

    assert definition.name == "specimen"
    assert definition.type == "cylinder"
    assert definition.parameters["radius"] == 0.005


@pytest.mark.parametrize("name", ["", "   "])
def test_geometry_definition_rejects_empty_name(name: str) -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        GeometryDefinition(
            name=name,
            type="cylinder",
            parameters={},
        )


@pytest.mark.parametrize("geometry_type", ["", "   "])
def test_geometry_definition_rejects_empty_type(
    geometry_type: str,
) -> None:
    with pytest.raises(ValueError, match="type must not be empty"):
        GeometryDefinition(
            name="specimen",
            type=geometry_type,
            parameters={},
        )


@pytest.mark.parametrize("name", [123, None])
def test_geometry_definition_rejects_non_string_name(
    name: object,
) -> None:
    with pytest.raises(TypeError, match="name must be a string"):
        GeometryDefinition(
            name=name,
            type="cylinder",
            parameters={},
        )


@pytest.mark.parametrize("geometry_type", [123, None])
def test_geometry_definition_rejects_non_string_type(
    geometry_type: object,
) -> None:
    with pytest.raises(TypeError, match="type must be a string"):
        GeometryDefinition(
            name="specimen",
            type=geometry_type,
            parameters={},
        )


@pytest.mark.parametrize("parameters", [None, [], "parameters", 1])
def test_geometry_definition_rejects_non_mapping_parameters(
    parameters: object,
) -> None:
    with pytest.raises(TypeError, match="parameters must be a mapping"):
        GeometryDefinition(
            name="specimen",
            type="cylinder",
            parameters=parameters,
        )


def test_geometry_definition_is_frozen() -> None:
    definition = GeometryDefinition(
        name="specimen",
        type="cylinder",
        parameters={},
    )

    with pytest.raises(AttributeError):
        definition.name = "other"
