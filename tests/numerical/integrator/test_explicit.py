"""Tests for explicit time integration."""

from __future__ import annotations

import pytest

from granular_rves.numerical.integrator.explicit import (
    ExplicitIntegrator,
    ExplicitState,
)


def test_explicit_integrator_requires_positive_dt() -> None:
    """An explicit integrator requires a positive time-step size."""
    with pytest.raises(ValueError, match="must be positive"):
        ExplicitIntegrator(dt=0.0)

    with pytest.raises(ValueError, match="must be positive"):
        ExplicitIntegrator(dt=-0.1)


def test_explicit_integrator_stores_dt() -> None:
    """The integrator stores the configured time-step size."""
    integrator = ExplicitIntegrator(dt=0.1)

    assert integrator.dt == 0.1


def test_explicit_step_updates_velocity_and_displacement() -> None:
    """A step updates velocity and displacement using acceleration."""
    integrator = ExplicitIntegrator(dt=0.1)

    state = ExplicitState(
        displacement=0.0,
        velocity=1.0,
    )

    def acceleration(state: ExplicitState, time: float) -> float:
        return 2.0

    result = integrator.step(
        state=state,
        time=0.0,
        acceleration=acceleration,
    )

    assert result.state.velocity == pytest.approx(1.2)
    assert result.state.displacement == pytest.approx(0.12)


def test_explicit_step_advances_time() -> None:
    """A step advances the simulation time by dt."""
    integrator = ExplicitIntegrator(dt=0.1)

    state = ExplicitState(
        displacement=0.0,
        velocity=0.0,
    )

    result = integrator.step(
        state=state,
        time=1.0,
        acceleration=lambda state, time: 0.0,
    )

    assert result.time == pytest.approx(1.1)
    assert result.dt == pytest.approx(0.1)


def test_explicit_step_is_accepted_and_complete() -> None:
    """A successful explicit step is accepted and complete."""
    integrator = ExplicitIntegrator(dt=0.1)

    state = ExplicitState(
        displacement=0.0,
        velocity=0.0,
    )

    result = integrator.step(
        state=state,
        time=0.0,
        acceleration=lambda state, time: 0.0,
    )

    assert result.accepted is True
    assert result.converged is True


def test_explicit_step_does_not_modify_previous_state() -> None:
    """A step returns a new state rather than modifying the input state."""
    integrator = ExplicitIntegrator(dt=0.1)

    state = ExplicitState(
        displacement=1.0,
        velocity=2.0,
    )

    result = integrator.step(
        state=state,
        time=0.0,
        acceleration=lambda state, time: 3.0,
    )

    assert state.displacement == 1.0
    assert state.velocity == 2.0
    assert result.state is not state


