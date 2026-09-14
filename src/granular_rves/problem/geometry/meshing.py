from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class BuildableGeometry(Protocol):
    """Interface required by geometry objects used for meshing."""

    def build(self, model: object) -> int:
        """Build the geometry in a Gmsh model and return its volume tag."""
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
    """Generate a three-dimensional DOLFINx mesh from geometry objects.

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
    Geometry objects are responsible for constructing their own geometric
    entities. This function is responsible for synchronizing the Gmsh
    geometry, applying the mesh discretization policy, generating the
    three-dimensional mesh, and converting the result to the DOLFINx
    representation.

    The current implementation generates tetrahedral volume meshes and
    does not yet assign semantic physical groups for individual boundary
    regions. That responsibility will be added when boundary and contact
    definitions are introduced.
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

        volume_tags = []

        for geometry_object in geometry:
            volume_tags.append(geometry_object.build(model))

        model.occ.synchronize()

        model.addPhysicalGroup(3, volume_tags, tag=1)
        model.setPhysicalName(3, 1, "volume")

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
