from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KinematicsDefinition:
    """Declarative description of the kinematics model.

    Parameters
    ----------
    type
        Name of the kinematics model, such as ``"small_strain"``.
        Whether the type is supported is checked by the kinematics
        registry.
    parameters
        Model-specific parameters required to define the kinematics.

    Raises
    ------
    TypeError
        If ``type`` is not a string or if ``parameters`` is not a mapping.
    ValueError
        If ``type`` is empty or contains only whitespace.
    """

    type: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate the generic kinematics declaration."""
        if not isinstance(self.type, str):
            raise TypeError("type must be a string.")

        if not self.type.strip():
            raise ValueError("type must not be empty.")

        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping.")


@dataclass(frozen=True)
class ConstitutiveDefinition:
    """Declarative description of a constitutive model.

    Parameters
    ----------
    type
        Name of the constitutive model, such as ``"linear_elastic"``.
        Whether the type is supported is checked by the constitutive
        registry.
    parameters
        Model-specific parameters required to define the constitutive
        model.

    Raises
    ------
    TypeError
        If ``type`` is not a string or if ``parameters`` is not a mapping.
    ValueError
        If ``type`` is empty or contains only whitespace.
    """

    type: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate the generic constitutive declaration."""
        if not isinstance(self.type, str):
            raise TypeError("type must be a string.")

        if not self.type.strip():
            raise ValueError("type must not be empty.")

        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping.")


@dataclass(frozen=True)
class BalanceDefinition:
    """Declarative description of the balance-law model.

    Parameters
    ----------
    type
        Name of the balance law, such as ``"momentum"``.
        Whether the type is supported is checked by the balance registry.
    parameters
        Balance-law-specific parameters.

    Raises
    ------
    TypeError
        If ``type`` is not a string or if ``parameters`` is not a mapping.
    ValueError
        If ``type`` is empty or contains only whitespace.
    """

    type: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate the generic balance-law declaration."""
        if not isinstance(self.type, str):
            raise TypeError("type must be a string.")

        if not self.type.strip():
            raise ValueError("type must not be empty.")

        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping.")


@dataclass(frozen=True)
class MechanicsDefinition:
    """Declarative description of the mechanics models in a problem.

    Parameters
    ----------
    kinematics
        Kinematics model used to describe deformation.
    constitutive
        Constitutive model relating kinematic quantities to stresses.
    balance
        Balance law governing mechanical equilibrium.

    Notes
    -----
    This class specifies which mechanics models are requested. It does not
    construct FEniCSx objects, assemble variational forms, or perform any
    numerical operations. Model construction is handled by the corresponding
    mechanics subpackages.
    """

    kinematics: KinematicsDefinition
    constitutive: ConstitutiveDefinition
    balance: BalanceDefinition
