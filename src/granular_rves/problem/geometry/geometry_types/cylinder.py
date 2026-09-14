from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence


@dataclass(frozen=True)
class Cylinder:
    """Three-dimensional cylindrical geometry.

    A cylinder is defined by two points on its axis and a positive radius.
    The two axis points determine both the cylinder position and orientation.

    Parameters
    ----------
    x0
        Three-dimensional coordinates of the first point on the cylinder
        axis.

    x1
        Three-dimensional coordinates of the second point on the cylinder
        axis.

    radius
        Positive cylinder radius.

    Raises
    ------
    ValueError
        If either axis point does not contain exactly three coordinates,
        contains non-finite values, if the radius is not positive and
        finite, or if the two axis points are identical.

    Notes
    -----
    This class represents the geometry of a cylinder. The :meth:`build`
    method creates the corresponding OpenCASCADE geometric entity in Gmsh,
    but does not generate a finite-element mesh or create a DOLFINx mesh.
    """

    x0: Sequence[float]
    x1: Sequence[float]
    radius: float

    def __post_init__(self) -> None:
        """Validate and normalize the cylinder geometry parameters."""
        if len(self.x0) != 3:
            raise ValueError(
                "x0 must contain exactly three coordinates."
            )

        if len(self.x1) != 3:
            raise ValueError(
                "x1 must contain exactly three coordinates."
            )

        x0 = tuple(float(value) for value in self.x0)
        x1 = tuple(float(value) for value in self.x1)
        radius = float(self.radius)

        if not all(isfinite(value) for value in x0):
            raise ValueError("x0 must contain only finite coordinates.")

        if not all(isfinite(value) for value in x1):
            raise ValueError("x1 must contain only finite coordinates.")

        if not isfinite(radius):
            raise ValueError("radius must be finite.")

        if radius <= 0.0:
            raise ValueError("radius must be positive.")

        if x0 == x1:
            raise ValueError(
                "x0 and x1 must define a non-zero cylinder axis."
            )

        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "radius", radius)

    def build(self, model: Any) -> int:
        """Create the cylinder in a Gmsh OpenCASCADE model.

        Parameters
        ----------
        model
            Gmsh model containing the OpenCASCADE geometry kernel.

        Returns
        -------
        int
            Gmsh volume tag for the constructed cylinder.

        Raises
        ------
        ValueError
            If the cylinder geometry has invalid parameters.

        Notes
        -----
        This method creates geometric CAD entities only. Mesh generation,
        physical-group assignment, and conversion to DOLFINx are handled by
        the meshing layer.
        """
        dx = self.x1[0] - self.x0[0]
        dy = self.x1[1] - self.x0[1]
        dz = self.x1[2] - self.x0[2]

        volume_tag = model.occ.addCylinder(
            self.x0[0],
            self.x0[1],
            self.x0[2],
            dx,
            dy,
            dz,
            self.radius,
        )

        return volume_tag
