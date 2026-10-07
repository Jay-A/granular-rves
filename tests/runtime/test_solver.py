from types import SimpleNamespace

import pytest

from granular_rves.runtime.solver import SolverController


def test_solver_controller_rejects_unsupported_analysis():
    problem = SimpleNamespace(analysis="unsupported")

    controller = SolverController(
        problem=problem,
        report=lambda message: None,
    )

    with pytest.raises(ValueError, match="Unsupported analysis type"):
        controller.solve()


def test_solver_controller_rejects_unimplemented_transient_analysis():
    problem = SimpleNamespace(analysis="transient")

    controller = SolverController(
        problem=problem,
        report=lambda message: None,
    )

    with pytest.raises(
        NotImplementedError,
        match="Transient analysis is not implemented yet",
    ):
        controller.solve()


def test_solver_controller_dispatches_steady_analysis():
    problem = SimpleNamespace(analysis="steady")

    controller = SolverController(
        problem=problem,
        report=lambda message: None,
    )

    expected = object()
    controller._solve_steady = lambda: expected

    assert controller.solve() is expected
