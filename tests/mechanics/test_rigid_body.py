"""Tests for rigid-body mechanics definitions."""

from granular_rves.mechanics.rigid_body import RigidBodyConstraintMode


def test_rigid_body_constraint_modes() -> None:
    """Supported rigid-body constraint modes have the expected values."""
    assert RigidBodyConstraintMode.UNCONSTRAINED.value == "unconstrained"
    assert RigidBodyConstraintMode.ZERO.value == "zero"
    assert RigidBodyConstraintMode.MEAN_ZERO.value == "mean_zero"


def test_rigid_body_constraint_mode_rejects_invalid_value() -> None:
    """Invalid rigid-body constraint modes are rejected."""
    import pytest

    with pytest.raises(ValueError):
        RigidBodyConstraintMode("invalid")
