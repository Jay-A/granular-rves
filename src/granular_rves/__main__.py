"""Command-line interface for granular-rves.

This module provides the entry point used by::

    python -m granular_rves <problem.yaml>

The command-line interface owns application-level orchestration and
diagnostic reporting. Numerical algorithms remain in the
:mod:`granular_rves.numerical` package.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from granular_rves.mechanics.constitutive.linear_elastic import LinearElastic
from granular_rves.mechanics.kinematics.small_strain import (
    SmallStrainKinematics,
)
from granular_rves.numerical.boundary.dirichlet import create_dirichlet_bcs
from granular_rves.numerical.boundary.reference import create_boundary_measure
from granular_rves.numerical.formulation.elasticity import ElasticityFormulation
from granular_rves.numerical.formulation.rigid_body import (
    RigidBodyFormulation,
)
from granular_rves.numerical.solver.steady import SteadySolver
from granular_rves.problem.geometry.geometry_types.cylinder import Cylinder
from granular_rves.problem.geometry.meshing import (
    MeshSettings,
    create_mesh,
)
from granular_rves.problem.loader import load_problem
from granular_rves.io.output import write_solution


class Reporter:
    """Application-level diagnostic reporter.

    Parameters
    ----------
    verbose
        If ``True``, diagnostic messages are emitted. If ``False``,
        reporting calls are ignored.

    Notes
    -----
    This class is intentionally small so that its implementation can
    later be replaced by a logger-backed reporter without coupling the
    numerical implementation to the command-line interface.
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the reporter.

        Parameters
        ----------
        verbose
            Enable diagnostic output when ``True``.
        """
        self.verbose = verbose

    def __call__(self, message: str) -> None:
        """Report a diagnostic message.

        Parameters
        ----------
        message
            Message to report.
        """
        if self.verbose:
            print(f"[granular-rves] {message}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for the granular-rves command-line interface.
    """
    parser = argparse.ArgumentParser(
        prog="granular-rves",
        description="Run a granular-rves simulation problem.",
    )
    parser.add_argument(
        "problem",
        help="Path to the YAML problem-definition file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose diagnostic output.",
    )
    return parser


def solve_steady(
    problem,
    report: Callable[[str], None],
):
    """Solve a steady-state problem.

    Parameters
    ----------
    problem
        Loaded declarative problem definition.
    report
        Application-level diagnostic reporting callable.

    Returns
    -------
    dolfinx.fem.Function
        Computed steady-state solution.
    """
    from mpi4py import MPI

    report("Building physical geometry.")

    geometry = Cylinder(
        x0=tuple(problem.geometry.parameters["x0"]),
        x1=tuple(problem.geometry.parameters["x1"]),
        radius=problem.geometry.parameters["radius"],
    )

    report("Generating mesh.")

    mesh_data = create_mesh(
        geometry=[geometry],
        settings=MeshSettings(
            characteristic_length=problem.mesh.size,
            order=problem.mesh.order,
        ),
        comm=MPI.COMM_WORLD,
    )

    report("Building mechanics formulation.")

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

    report("Applying Dirichlet boundary conditions.")

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
        report("Building rigid-body constraints.")

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

    report("Creating steady-state solver.")

    solver = SteadySolver(
        formulation=formulation,
        boundary_conditions=boundary_conditions,
        constraints=constraints,
    )

    report("Starting steady-state solve.")

    solution = solver.solve()

    report("Steady-state solve completed.")


    write_solution(
        mesh_data=mesh_data,
        solution=solution,
        output_directory=problem.output.directory,
    )

    report("Solution written to output.")

    return solution


def main() -> None:
    """Run the granular-rves command-line interface."""
    parser = build_parser()
    args = parser.parse_args()

    report = Reporter(verbose=args.verbose)

    report("Loading problem definition.")

    problem = load_problem(args.problem)

    report(
        f"Loaded problem {problem.name!r} "
        f"with analysis type {problem.analysis!r}."
    )

    if problem.analysis == "steady":
        solve_steady(
            problem,
            report,
        )
    elif problem.analysis == "transient":
        raise NotImplementedError(
            "Transient analysis is not implemented yet.",
        )
    else:
        raise ValueError(
            f"Unsupported analysis type: {problem.analysis!r}.",
        )


if __name__ == "__main__":
    main()


