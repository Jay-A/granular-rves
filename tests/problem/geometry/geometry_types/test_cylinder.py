import pytest

from granular_rves.problem.geometry.geometry_types.cylinder import Cylinder


def test_valid_cylinder() -> None:
    cylinder = Cylinder(
        x0=[0.0, 0.0, 0.0],
        x1=[0.0, 0.0, 0.01],
        radius=0.005,
    )

    assert cylinder.x0 == (0.0, 0.0, 0.0)
    assert cylinder.x1 == (0.0, 0.0, 0.01)
    assert cylinder.radius == 0.005


@pytest.mark.parametrize(
    "x0",
    [
        [0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
)
def test_cylinder_rejects_invalid_x0(x0: list[float]) -> None:
    with pytest.raises(
        ValueError,
        match="x0 must contain exactly three coordinates",
    ):
        Cylinder(
            x0=x0,
            x1=[0.0, 0.0, 0.01],
            radius=0.005,
        )


@pytest.mark.parametrize(
    "x1",
    [
        [0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
)
def test_cylinder_rejects_invalid_x1(x1: list[float]) -> None:
    with pytest.raises(
        ValueError,
        match="x1 must contain exactly three coordinates",
    ):
        Cylinder(
            x0=[0.0, 0.0, 0.0],
            x1=x1,
            radius=0.005,
        )


@pytest.mark.parametrize(
    "radius",
    [0.0, -1.0],
)
def test_cylinder_rejects_non_positive_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="radius must be positive"):
        Cylinder(
            x0=[0.0, 0.0, 0.0],
            x1=[0.0, 0.0, 0.01],
            radius=radius,
        )


@pytest.mark.parametrize(
    "radius",
    [float("inf"), float("-inf"), float("nan")],
)
def test_cylinder_rejects_non_finite_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="radius must be finite"):
        Cylinder(
            x0=[0.0, 0.0, 0.0],
            x1=[0.0, 0.0, 0.01],
            radius=radius,
        )


def test_cylinder_rejects_identical_axis_points() -> None:
    with pytest.raises(
        ValueError,
        match="x0 and x1 must define a non-zero cylinder axis",
    ):
        Cylinder(
            x0=[0.0, 0.0, 0.0],
            x1=[0.0, 0.0, 0.0],
            radius=0.005,
        )


@pytest.mark.parametrize(
    "point_name",
    ["x0", "x1"],
)
def test_cylinder_rejects_non_finite_coordinates(
    point_name: str,
) -> None:
    point = [0.0, 0.0, 0.01]

    if point_name == "x0":
        point[1] = float("nan")
        x0 = point
        x1 = [0.0, 0.0, 0.01]
    else:
        point[1] = float("inf")
        x0 = [0.0, 0.0, 0.0]
        x1 = point

    with pytest.raises(
        ValueError,
        match=f"{point_name} must contain only finite coordinates",
    ):
        Cylinder(
            x0=x0,
            x1=x1,
            radius=0.005,
        )


def test_cylinder_accepts_non_vertical_axis() -> None:
    cylinder = Cylinder(
        x0=[0.0, 0.0, 0.0],
        x1=[0.01, 0.02, 0.03],
        radius=0.005,
    )

    assert cylinder.x0 == (0.0, 0.0, 0.0)
    assert cylinder.x1 == (0.01, 0.02, 0.03)


def test_cylinder_is_frozen() -> None:
    cylinder = Cylinder(
        x0=[0.0, 0.0, 0.0],
        x1=[0.0, 0.0, 0.01],
        radius=0.005,
    )

    with pytest.raises(AttributeError):
        cylinder.radius = 0.01
