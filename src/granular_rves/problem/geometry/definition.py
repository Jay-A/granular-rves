"""Declarative definitions for physical geometry objects.

This module defines the problem-level representation of geometry
independently of concrete geometry construction and mesh generation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeometryDefinition:
    """Declarative description of a physical geometry object.

    Parameters
    ----------
    type
        Name of the geometry type, such as ``"cylinder"``.
    parameters
        Geometry-specific parameters required to construct the object.
    name
        Optional semantic name identifying the geometry object.
        If omitted, the geometry type is used as the default name.

    Raises
    ------
    TypeError
        If ``name`` or ``type`` is not a string, or if ``parameters`` is
        not a mapping.
    ValueError
        If ``name`` or ``type`` is empty or contains only whitespace.
    """

    type: str
    parameters: Mapping[str, Any]
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the geometry declaration."""
        if not isinstance(self.type, str):
            raise TypeError("type must be a string.")

        if not self.type.strip():
            raise ValueError("type must not be empty.")

        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping.")

        if self.name is None:
            object.__setattr__(self, "name", self.type)
        elif not isinstance(self.name, str):
            raise TypeError("name must be a string.")

        if not self.name.strip():
            raise ValueError("name must not be empty.")


