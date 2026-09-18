from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

from granular_rves.problem.geometry.meshing import GeometryEntities


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
    method creates the corresponding OpenCASCADE geometric entities in
    Gmsh, but does not generate a finite-element mesh or create a
    DOLFINx mesh.
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
            raise ValueError(
                "x0 must contain only finite coordinates."
            )

        if not all(isfinite(value) for value in x1):
            raise ValueError(
                "x1 must contain only finite coordinates."
            )

        if not isfinite(radius):
            raise ValueError(
                "radius must be finite."
            )

        if radius <= 0.0:
            raise ValueError(
                "radius must be positive."
            )

        if x0 == x1:
            raise ValueError(
                "x0 and x1 must define a non-zero cylinder axis."
            )

        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "radius", radius)

    def build(self, model: Any) -> GeometryEntities:
        """Create the cylinder in a Gmsh OpenCASCADE model.

        Parameters
        ----------
        model
            Gmsh model containing the OpenCASCADE geometry kernel.

        Returns
        -------
        GeometryEntities
            Gmsh volume and boundary-surface entities created by the
            cylinder. The surfaces are identified as ``"top"``,
            ``"bottom"``, and ``"lateral"``.

        Raises
        ------
        ValueError
            If the cylinder geometry has invalid parameters or if the
            generated CAD boundary cannot be identified as one lateral
            surface and two end surfaces.

        Notes
        -----
        This method creates geometric CAD entities only. Mesh generation,
        physical-group assignment, and conversion to DOLFINx are handled
        by the meshing layer.

        The boundary surfaces are identified from the OpenCASCADE
        geometry rather than assuming particular Gmsh entity tags.
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

        model.occ.synchronize()

        boundary = model.getBoundary(
            [(3, volume_tag)],
            combined=False,
            oriented=False,
        )

        surface_tags = [
            tag
            for dim, tag in boundary
            if dim == 2
        ]

        if len(surface_tags) != 3:
            raise ValueError(
                "Expected a cylinder to have exactly three boundary "
                f"surfaces, found {len(surface_tags)}."
            )

        axis_length = (
            dx * dx
            + dy * dy
            + dz * dz
        ) ** 0.5

        axis = (
            dx / axis_length,
            dy / axis_length,
            dz / axis_length,
        )

        tol = 1.0e-8

        end_surfaces: dict[str, int] = {}
        lateral_surfaces: list[int] = []

        for surface_tag in surface_tags:
            bbox = model.getBoundingBox(2, surface_tag)

            center = (
                0.5 * (bbox[0] + bbox[3]),
                0.5 * (bbox[1] + bbox[4]),
                0.5 * (bbox[2] + bbox[5]),
            )

            relative = (
                center[0] - self.x0[0],
                center[1] - self.x0[1],
                center[2] - self.x0[2],
            )

            projection = (
                relative[0] * axis[0]
                + relative[1] * axis[1]
                + relative[2] * axis[2]
            )

            if abs(projection) <= tol:
                end_surfaces["bottom"] = surface_tag
            elif abs(projection - axis_length) <= tol:
                end_surfaces["top"] = surface_tag
            else:
                lateral_surfaces.append(surface_tag)

        if len(end_surfaces) != 2:
            raise ValueError(
                "Could not identify the cylinder's top and bottom "
                "surfaces."
            )

        if len(lateral_surfaces) != 1:
            raise ValueError(
                "Could not identify the cylinder's lateral surface."
            )

        return GeometryEntities(
            volume=volume_tag,
            surfaces={
                "top": end_surfaces["top"],
                "bottom": end_surfaces["bottom"],
                "lateral": lateral_surfaces[0],
            },
        )


