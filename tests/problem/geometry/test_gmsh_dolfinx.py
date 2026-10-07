from __future__ import annotations

import gmsh
from mpi4py import MPI

from granular_rves.problem.geometry.geometry_types.cylinder import Cylinder


def test_gmsh_dolfinx_cylinder() -> None:
    """Test Gmsh cylinder generation and entity creation."""
    gmsh.initialize()

    try:
        model = gmsh.model
        model.add("test_cylinder")

        geometry = Cylinder(
            x0=(0.0, 0.0, 0.0),
            x1=(0.0, 0.0, 2.0),
            radius=1.0,
        )

        entities = geometry.build(model)

        model.occ.synchronize()

        # The cylinder should provide one volume and three named
        # boundary surfaces.
        assert entities.volume > 0

        assert set(entities.surfaces) == {
            "top",
            "bottom",
            "lateral",
        }

        assert all(
            tag > 0
            for tag in entities.surfaces.values()
        )

        # Verify that the returned volume and surfaces actually exist
        # in the Gmsh model.
        volume_entities = model.getEntities(3)
        surface_entities = model.getEntities(2)

        assert (3, entities.volume) in volume_entities

        for surface_tag in entities.surfaces.values():
            assert (2, surface_tag) in surface_entities

        # Generate a raw Gmsh mesh to verify that the returned CAD
        # entities are valid for three-dimensional meshing.
        model.mesh.setSize(
            model.getEntities(0),
            0.25,
        )
        model.mesh.generate(3)

        volume_elements = model.mesh.getElements(3)

        assert volume_elements[1]
        assert any(
            len(element_tags) > 0
            for element_tags in volume_elements[1]
        )

    finally:
        gmsh.finalize()


