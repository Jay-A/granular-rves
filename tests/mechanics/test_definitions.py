import pytest

from granular_rves.mechanics.definitions import (
    BalanceDefinition,
    ConstitutiveDefinition,
    KinematicsDefinition,
)


@pytest.mark.parametrize(
    "definition_type",
    [
        KinematicsDefinition,
        ConstitutiveDefinition,
        BalanceDefinition,
    ],
)
def test_definition_rejects_non_string_type(definition_type) -> None:
    with pytest.raises(TypeError, match="type must be a string"):
        definition_type(
            type=123,
            parameters={},
        )


@pytest.mark.parametrize(
    "definition_type",
    [
        KinematicsDefinition,
        ConstitutiveDefinition,
        BalanceDefinition,
    ],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_definition_rejects_empty_type(definition_type, value) -> None:
    with pytest.raises(ValueError, match="type must not be empty"):
        definition_type(
            type=value,
            parameters={},
        )


@pytest.mark.parametrize(
    "definition_type",
    [
        KinematicsDefinition,
        ConstitutiveDefinition,
        BalanceDefinition,
    ],
)
def test_definition_rejects_non_mapping_parameters(definition_type) -> None:
    with pytest.raises(TypeError, match="parameters must be a mapping"):
        definition_type(
            type="example",
            parameters=[],
        )


@pytest.mark.parametrize(
    "definition_type",
    [
        KinematicsDefinition,
        ConstitutiveDefinition,
        BalanceDefinition,
    ],
)
def test_definition_accepts_mapping(definition_type) -> None:
    parameters = {"example": 1.0}

    definition = definition_type(
        type="example",
        parameters=parameters,
    )

    assert definition.parameters == parameters
