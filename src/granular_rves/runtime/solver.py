"""Application-level solver control."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mpi4py import MPI


@dataclass(frozen=True)
class SolverResult:
    """Application-facing result of a numerical solve."""

    mesh_data: Any
    solution: Any
    constraint_rows: list[Any] | None = None
    constraint_multipliers: Any = None
    reaction_history: list[dict[str, float | int]] | None = None


from granular_rves.mechanics.constitutive.linear_elastic import LinearElastic
from granular_rves.mechanics.kinematics.small_strain import SmallStrainKinematics
from granular_rves.numerical.boundary.dirichlet import create_dirichlet_bcs
from granular_rves.numerical.boundary.reference import create_boundary_measure
from granular_rves.numerical.formulation.elasticity import ElasticityFormulation
from granular_rves.numerical.formulation.rigid_body import RigidBodyFormulation
from granular_rves.numerical.solver.steady import SteadySolver
from granular_rves.problem.geometry.geometry_types.cylinder import Cylinder
from granular_rves.problem.geometry.meshing import MeshSettings, create_mesh


class SolverController:
    """Control numerical solver execution for a simulation problem."""

    def __init__(
        self,
        problem: Any,
        report: Callable[[str], None],
    ) -> None:
        self.problem = problem
        self.report = report

    def solve(self) -> Any:
        """Solve the configured simulation problem."""
        if self.problem.analysis == "steady":
            return self._solve_steady()

        if self.problem.analysis == "transient":
            raise NotImplementedError(
                "Transient analysis is not implemented yet.",
            )

        raise ValueError(
            f"Unsupported analysis type: {self.problem.analysis!r}.",
        )

    def _solve_steady(self) -> Any:
        """Build and solve the current steady-state problem."""

        problem = self.problem

        self.report("Building physical geometry.")

        geometry = Cylinder(
            x0=tuple(problem.geometry.parameters["x0"]),
            x1=tuple(problem.geometry.parameters["x1"]),
            radius=problem.geometry.parameters["radius"],
        )

        self.report("Generating mesh.")

        mesh_data = create_mesh(
            geometry=[geometry],
            settings=MeshSettings(
                characteristic_length=problem.mesh.size,
                order=problem.mesh.order,
            ),
            comm=MPI.COMM_WORLD,
        )

        self.report("Building mechanics formulation.")

        kinematics = SmallStrainKinematics()

        constitutive_parameters = (
            problem.mechanics.constitutive.models["linear_elastic"]
        )

        constitutive = LinearElastic(
            youngs_modulus=constitutive_parameters["youngs_modulus"],
            poisson_ratio=constitutive_parameters["poisson_ratio"],
        )

        formulation = ElasticityFormulation(
            mesh_data=mesh_data,
            kinematics=kinematics,
            constitutive=constitutive,
        )

        self.report("Applying Dirichlet boundary conditions.")

        boundary_conditions = create_dirichlet_bcs(
            mesh_data=mesh_data,
            function_space=formulation.function_space,
            boundary=problem.boundary,
            values=(
                {
                    problem.loading.boundary: problem.loading.value,
                }
                if problem.loading is not None
                else None
            ),
        )

        constraints_definition = problem.constraints

        if constraints_definition.reference_boundary is not None:
            self.report("Building rigid-body constraints.")

            reference_measure = create_boundary_measure(
                mesh_data=mesh_data,
                boundary=problem.boundary,
                boundary_name=constraints_definition.reference_boundary,
            )

            rigid_body = RigidBodyFormulation(
                function_space=formulation.function_space,
                reference_measure=reference_measure,
            )

            constraints = rigid_body.constraints(
                constraints_definition,
            )
        else:
            constraints = []

        self.report("Creating steady-state solver.")

        solver = SteadySolver(
            formulation=formulation,
            boundary_conditions=boundary_conditions,
            constraints=constraints,
            boundary=problem.boundary,
            loading=problem.loading,
            reactions=problem.output.reactions,
        )

        self.report("Starting steady-state solve.")

        solution = solver.solve()

        self.report("Steady-state solve completed.")

        return SolverResult(
            mesh_data=mesh_data,
            solution=solution,
            constraint_rows=solver.constraint_rows,
            constraint_multipliers=solver.constraint_multipliers,
            reaction_history=solver.reaction_history,
        )

