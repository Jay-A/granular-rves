from __future__ import annotations

import gmsh
from mpi4py import MPI

from dolfinx.io import gmsh as dolfinx_gmsh


class Cylinder:
    """Minimal Gmsh cylinder implementing BuildableGeometry."""

    def __init__(self, radius: float, height: float):
        self.radius = radius
        self.height = height

    def build(self, model: object) -> int:
        tag = model.occ.addCylinder(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            self.height,
            self.radius,
        )
        return tag


def main() -> None:
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print("Initializing Gmsh...")

    gmsh.initialize()

    try:
        model = gmsh.model
        model.add("test_cylinder")

        geometry = Cylinder(radius=1.0, height=2.0)

        volume_tag = geometry.build(model)

        print(f"[rank {comm.rank}] Gmsh volume tag: {volume_tag}")

        model.occ.synchronize()

        # Coarse mesh for the first integration test.
        model.mesh.setSize(
            model.getEntities(0),
            0.25,
        )

        model.mesh.generate(3)

        if comm.rank == 0:
            print("Gmsh 3-D mesh generated.")

        mesh_data = dolfinx_gmsh.model_to_mesh(
            model,
            comm,
            rank=0,
        )

        msh = mesh_data.mesh

        if comm.rank == 0:
            print("DOLFINx mesh imported successfully.")
            print(f"Geometric dimension: {msh.geometry.dim}")
            print(f"Topological dimension: {msh.topology.dim}")

        num_cells = msh.topology.index_map(
            msh.topology.dim
        ).size_local

        num_vertices = msh.topology.index_map(
            0
        ).size_local

        print(
            f"[rank {comm.rank}] "
            f"cells={num_cells}, vertices={num_vertices}"
        )

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
