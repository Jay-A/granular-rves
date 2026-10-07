from types import SimpleNamespace

from granular_rves.runtime.solver import SolverResult
from granular_rves.runtime.simulation import SimulationManager


def test_simulation_manager_runs_solver_and_returns_result():
    problem = SimpleNamespace(
        name="test-problem",
        analysis="steady",
        output=SimpleNamespace(
            directory="output",
            reaction_history=True,
        ),
    )

    messages = []
    mesh_data = object()
    solution = object()
    reaction_history = [
        {
            "name": "top_force",
            "step": 0,
            "time": 0.0,
            "reaction": 0.0,
        },
        {
            "name": "top_force",
            "step": 1,
            "time": 1.0,
            "reaction": -123.0,
        },
    ]

    result = SolverResult(
        mesh_data=mesh_data,
        solution=solution,
        reaction_history=reaction_history,
    )

    class FakeSolverController:
        def __init__(self, problem, report):
            assert problem is problem_definition
            assert report is report_function

        def solve(self):
            return result

    problem_definition = problem
    report_function = messages.append

    import granular_rves.runtime.simulation as simulation_module

    original_controller = simulation_module.SolverController
    original_writer = simulation_module.write_solution
    original_reaction_writer = simulation_module.write_reaction_history

    simulation_module.SolverController = FakeSolverController

    recorded = {}

    def fake_write_solution(mesh_data, solution, output_directory):
        recorded["mesh_data"] = mesh_data
        recorded["solution"] = solution
        recorded["output_directory"] = output_directory

    simulation_module.write_solution = fake_write_solution

    def fake_write_reaction_history(reaction_history, output_directory):
        recorded["reaction_history"] = reaction_history
        recorded["reaction_output_directory"] = output_directory

    simulation_module.write_reaction_history = fake_write_reaction_history

    try:
        manager = SimulationManager(
            problem=problem,
            report=report_function,
        )

        assert manager.run() is result
    finally:
        simulation_module.SolverController = original_controller
        simulation_module.write_solution = original_writer
        simulation_module.write_reaction_history = original_reaction_writer

    assert recorded == {
        "mesh_data": mesh_data,
        "solution": solution,
        "output_directory": "output",
        "reaction_history": reaction_history,
        "reaction_output_directory": "output",
    }

    assert messages == [
        "Starting simulation 'test-problem' with analysis type 'steady'.",
        "Solving simulation.",
        "Recording simulation solution.",
        "Simulation completed.",
    ]
