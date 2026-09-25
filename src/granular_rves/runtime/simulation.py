"""Application-level simulation orchestration."""

from __future__ import annotations

from typing import Any

from granular_rves.io.output import write_solution, write_reaction_history
from granular_rves.runtime.solver import SolverController


class SimulationManager:
    """Coordinate execution of a granular-rves simulation."""

    def __init__(
        self,
        problem: Any,
        report: Any,
    ) -> None:
        self.problem = problem
        self.report = report

    def run(self) -> Any:
        """Run the configured simulation."""

        self.report(
            f"Starting simulation {self.problem.name!r} "
            f"with analysis type {self.problem.analysis!r}."
        )

        solver = SolverController(
            problem=self.problem,
            report=self.report,
        )

        self.report("Solving simulation.")

        result = solver.solve()

        self.report("Recording simulation solution.")

        write_solution(
            mesh_data=result.mesh_data,
            solution=result.solution,
            output_directory=self.problem.output.directory,
        )

        if self.problem.output.reaction_history:
            write_reaction_history(
                reaction_history=result.reaction_history,
                output_directory=self.problem.output.directory,
            )

        self.report("Simulation completed.")

        return result
