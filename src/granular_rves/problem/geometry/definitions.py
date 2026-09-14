from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeometryDefinition:
    """Declarative description of a physical geometry object.

    A geometry definition specifies what physical object is requested without
    specifying how that object is constructed or meshed. Geometry-specific
    validation belongs to the corresponding geometry type.

    Parameters
    ----------
    name
        Semantic name identifying the geometry object within the problem.

    type
        Name of the geometry type, such as ``"cylinder"``. Whether the type
        is actually supported is checked by the geometry registry.

    parameters
        Geometry-specific parameters required to define the physical object.

    Raises
    ------
    TypeError
        If ``name`` or ``type`` is not a string, or if ``parameters`` is not
        a mapping.

    ValueError
        If ``name`` or ``type`` is empty or contains only whitespace.
    """

    name: str
    type: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate the generic geometry declaration."""
        if not isinstance(self.name, str):
            raise TypeError("name must be a string.")

        if not self.name.strip():
            raise ValueError("name must not be empty.")

        if not isinstance(self.type, str):
            raise TypeError("type must be a string.")

        if not self.type.strip():
            raise ValueError("type must not be empty.")

        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping.")
