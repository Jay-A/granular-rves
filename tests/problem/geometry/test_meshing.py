import pytest

from granular_rves.problem.geometry.geometry_types.cylinder import Cylinder
from granular_rves.problem.geometry.meshing import (
    MeshSettings,
    create_mesh,
)


def test_mesh_settings_defaults() -> None:
    settings = MeshSettings(characteristic_length=0.001)

    assert settings.characteristic_length == 0.001
    assert settings.order == 1


@pytest.mark.parametrize(
    "characteristic_length",
    [0.0, -1.0],
)
def test_mesh_settings_rejects_non_positive_characteristic_length(
    characteristic_length: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="characteristic_length must be positive",
    ):
        MeshSettings(
            characteristic_length=characteristic_length,
        )


@pytest.mark.parametrize("order", [0, -1])
def test_mesh_settings_rejects_invalid_order(order: int) -> None:
    with pytest.raises(
        ValueError,
        match="order must be at least 1",
    ):
        MeshSettings(
            characteristic_length=0.001,
            order=order,
        )


def test_create_mesh_rejects_empty_geometry() -> None:
    settings = MeshSettings(characteristic_length=0.002)

    with pytest.raises(
        ValueError,
        match="At least one geometry object is required",
    ):
        create_mesh([], settings)


def test_create_mesh_from_cylinder() -> None:
    cylinder = Cylinder(
        x0=[0.0, 0.0, 0.0],
        x1=[0.0, 0.0, 0.01],
        radius=0.005,
    )

    settings = MeshSettings(
        characteristic_length=0.002,
    )

    mesh_data = create_mesh(
        [cylinder],
        settings,
    )

    assert mesh_data.mesh.geometry.dim == 3
    assert mesh_data.mesh.topology.dim == 3
    assert mesh_data.mesh.topology.index_map(3).size_local > 0
