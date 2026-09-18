"""Explicit time integration for second-order mechanical systems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExplicitState:
    """State of a second-order mechanical system.

    Parameters
    ----------
    displacement
        Displacement state at the current time.
    velocity
        Velocity state at the current time.
    """

    displacement: Any
    velocity: Any


@dataclass(frozen=True)
class ExplicitStepResult:
    """Result of one explicit integration step.

    Parameters
    ----------
    state
        State at the end of the integration step.
    time
        Time at the end of the integration step.
    dt
        Time-step size used for the integration step.
    accepted
        Whether the step was accepted.
    converged
        Whether the step satisfied the integrator's completion criteria.

    Notes
    -----
    Explicit integration does not require an iterative convergence
    procedure in the same sense as an implicit method. Consequently,
    an explicit step is normally both accepted and complete when the
    update succeeds.
    """

    state: ExplicitState
    time: float
    dt: float
    accepted: bool
    converged: bool


class ExplicitIntegrator:
    """Advance a second-order mechanical state explicitly in time.

    The integrator owns the time-stepping operation but does not know how
    acceleration is obtained. The acceleration is supplied through a
    callback, allowing the numerical mechanics and solver layers to
    provide the physical update without coupling this class to a
    particular finite-element implementation.

    The current implementation uses a forward-Euler update,

    .. math::

        \\mathbf{v}_{n+1}
        =
        \\mathbf{v}_n + \\Delta t\\,\\mathbf{a}_n,

    .. math::

        \\mathbf{u}_{n+1}
        =
        \\mathbf{u}_n + \\Delta t\\,\\mathbf{v}_{n+1}.

    This is intentionally a minimal first explicit integrator. A
    production dynamics implementation may replace this update with a
    more appropriate explicit scheme, such as central difference,
    without changing the surrounding solver architecture.

    Parameters
    ----------
    dt
        Time-step size. Must be positive.

    Raises
    ------
    ValueError
        If ``dt`` is not positive.
    """

    def __init__(self, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("Time-step size dt must be positive.")

        self.dt = dt

    def step(
        self,
        state: ExplicitState,
        time: float,
        acceleration: Callable[[ExplicitState, float], Any],
    ) -> ExplicitStepResult:
        """Advance the state by one explicit time step.

        Parameters
        ----------
        state
            State at the beginning of the time step.
        time
            Current simulation time.
        acceleration
            Callable returning the acceleration for the current state.
            The callable receives ``(state, time)``.

        Returns
        -------
        ExplicitStepResult
            State and metadata for the completed time step.
        """
        acceleration_value = acceleration(state, time)

        velocity = state.velocity + self.dt * acceleration_value
        displacement = state.displacement + self.dt * velocity

        next_state = ExplicitState(
            displacement=displacement,
            velocity=velocity,
        )

        return ExplicitStepResult(
            state=next_state,
            time=time + self.dt,
            dt=self.dt,
            accepted=True,
            converged=True,
        )
