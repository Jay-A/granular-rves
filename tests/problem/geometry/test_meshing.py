from __future__ import annotations

from granular_rves.problem.geometry.geometry_types.cylinder import Cylinder
from granular_rves.problem.geometry.meshing import MeshSettings, create_mesh


def test_create_mesh_cylinder() -> None:
    """Test creation of a DOLFINx mesh from a cylinder."""
    geometry = Cylinder(
        x0=(0.0, 0.0, 0.0),
        x1=(0.0, 0.0, 2.0),
        radius=1.0,
    )

    settings = MeshSettings(
        characteristic_length=0.25,
    )

    mesh_data = create_mesh(
        [geometry],
        settings,
    )

    mesh = mesh_data.mesh

    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 3

    num_cells = mesh.topology.index_map(
        mesh.topology.dim
    ).size_local

    num_vertices = mesh.topology.index_map(
        0
    ).size_local

    assert num_cells > 0
    assert num_vertices > 0

    # Verify that the geometry boundary regions were transferred
    # from Gmsh into the DOLFINx mesh metadata.
    assert mesh_data.facet_tags is not None

    physical_groups = mesh_data.physical_groups

    assert "volume" in physical_groups
    assert "top" in physical_groups
    assert "bottom" in physical_groups
    assert "lateral" in physical_groups

    assert physical_groups["volume"].dim == 3
    assert physical_groups["top"].dim == 2
    assert physical_groups["bottom"].dim == 2
    assert physical_groups["lateral"].dim == 2

    facet_tags = mesh_data.facet_tags

    top_tag = physical_groups["top"].tag
    bottom_tag = physical_groups["bottom"].tag
    lateral_tag = physical_groups["lateral"].tag

    assert facet_tags.find(top_tag).size > 0
    assert facet_tags.find(bottom_tag).size > 0
    assert facet_tags.find(lateral_tag).size > 0
