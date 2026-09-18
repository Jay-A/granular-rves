from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GeometryEntities:
    """Gmsh entities created by a geometry object.

    Parameters
    ----------
    volume
        Gmsh tag identifying the three-dimensional volume.
    surfaces
        Mapping from geometry-specific boundary names to Gmsh surface
        tags.

    Notes
    -----
    The names in ``surfaces`` describe geometric regions created by the
    geometry object. They do not represent mechanical boundary
    conditions or contact definitions.
    """

    volume: int
    surfaces: dict[str, int]


class BuildableGeometry(Protocol):
    """Interface required by geometry objects used for meshing."""

    def build(self, model: object) -> GeometryEntities:
        """Build the geometry in a Gmsh model.

        Parameters
        ----------
        model
            Gmsh model containing the OpenCASCADE geometry kernel.

        Returns
        -------
        GeometryEntities
            Gmsh volume and boundary-surface entities created by the
            geometry object.
        """
        ...


@dataclass(frozen=True)
class MeshSettings:
    """Settings controlling finite-element mesh generation.

    Parameters
    ----------
    characteristic_length
        Target Gmsh mesh size used for the initial discretization.
    order
        Polynomial order of the generated finite-element mesh.
        The initial project baseline uses first-order elements.
    """

    characteristic_length: float
    order: int = 1

    def __post_init__(self) -> None:
        """Validate mesh settings."""
        if self.characteristic_length <= 0.0:
            raise ValueError(
                "characteristic_length must be positive."
            )

        if self.order < 1:
            raise ValueError(
                "order must be at least 1."
            )


def create_mesh(
    geometry: Sequence[BuildableGeometry],
    settings: MeshSettings,
    comm=None,
):
    """Generate a three-dimensional DOLFINx mesh from geometry.

    Parameters
    ----------
    geometry
        Geometry objects to construct in the Gmsh model.
    settings
        Mesh discretization settings.
    comm
        MPI communicator used to construct the DOLFINx mesh.

    Returns
    -------
    dolfinx.io.gmsh.MeshData
        DOLFINx mesh together with the associated Gmsh entity tags and
        physical-group information.

    Raises
    ------
    ValueError
        If no geometry objects are supplied.

    Notes
    -----
    Geometry objects are responsible for constructing their own
    geometric entities and identifying their geometric boundary
    regions. This function is responsible for assigning Gmsh physical
    groups, synchronizing the Gmsh geometry, applying the mesh
    discretization policy, generating the three-dimensional mesh, and
    converting the result to the DOLFINx representation.

    The generated volume is assigned the physical group ``"volume"``.
    Boundary surfaces are assigned physical groups using the names
    provided by each geometry object.

    The physical-group names describe geometric regions only. They do
    not prescribe mechanical boundary conditions, contact behavior, or
    loading.
    """
    if not geometry:
        raise ValueError("At least one geometry object is required.")

    import gmsh
    from dolfinx.io import gmsh as dolfinx_gmsh
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD

    gmsh.initialize()

    try:
        model = gmsh.model
        model.add("granular_rves")

        geometry_entities = [
            geometry_object.build(model)
            for geometry_object in geometry
        ]

        model.occ.synchronize()

        volume_tags = [
            entities.volume
            for entities in geometry_entities
        ]

        model.addPhysicalGroup(
            3,
            volume_tags,
            tag=1,
        )
        model.setPhysicalName(
            3,
            1,
            "volume",
        )

        next_physical_tag = 2

        for entities in geometry_entities:
            for name, surface_tag in entities.surfaces.items():
                model.addPhysicalGroup(
                    2,
                    [surface_tag],
                    tag=next_physical_tag,
                )
                model.setPhysicalName(
                    2,
                    next_physical_tag,
                    name,
                )
                next_physical_tag += 1

        model.mesh.setSize(
            model.getEntities(0),
            settings.characteristic_length,
        )

        model.mesh.generate(3)

        if settings.order > 1:
            model.mesh.setOrder(settings.order)

        return dolfinx_gmsh.model_to_mesh(
            model,
            comm,
            rank=0,
        )

    finally:
        gmsh.finalize()
