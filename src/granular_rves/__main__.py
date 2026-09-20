"""Command-line interface for granular-rves.

This module provides the entry point used by::

    python -m granular_rves <problem.yaml>

The command-line interface is intentionally kept thin. Numerical
algorithms remain in the :mod:`granular_rves.numerical` package.
"""

from __future__ import annotations

import argparse

from granular_rves.mechanics.constitutive.linear_elastic import LinearElastic
from granular_rves.mechanics.kinematics.small_strain import (
    SmallStrainKinematics,
)
from granular_rves.mechanics.rigid_body import RigidBodyConstraintMode
from granular_rves.numerical.boundary.dirichlet import create_dirichlet_bcs
from granular_rves.numerical.boundary.reference import create_boundary_measure
from granular_rves.numerical.formulation.elasticity import ElasticityFormulation
from granular_rves.numerical.formulation.rigid_body import (
    RigidBodyFormulation,
)
from granular_rves.numerical.solver.steady import SteadySolver
from granular_rves.problem.definition import RigidBodyConstraintDefinition
from granular_rves.problem.geometry.geometry_types.cylinder import Cylinder
from granular_rves.problem.geometry.meshing import (
    MeshSettings,
    create_mesh,
)
from granular_rves.problem.loader import load_problem


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

    return parser


def solve_steady(problem) -> None:
    """Solve a steady-state problem.

    Parameters
    ----------
    problem
        Loaded declarative problem definition.
    """
    from mpi4py import MPI

    geometry = Cylinder(
        x0=tuple(problem.geometry.parameters["x0"]),
        x1=tuple(problem.geometry.parameters["x1"]),
        radius=problem.geometry.parameters["radius"],
    )

    mesh_data = create_mesh(
        geometry=[geometry],
        settings=MeshSettings(
            characteristic_length=problem.mesh.size,
            order=problem.mesh.order,
        ),
        comm=MPI.COMM_WORLD,
    )

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
        reference_measure = create_boundary_measure(
            mesh_data=mesh_data,
            boundary=problem.boundary,
            boundary_name=constraints_definition.reference_boundary,
        )

        rigid_body = RigidBodyFormulation(
            function_space=formulation.function_space,
            reference_measure=reference_measure,
        )

        constraints = rigid_body.constraints(constraints_definition)
    else:
        constraints = []

    solver = SteadySolver(
        formulation=formulation,
        boundary_conditions=boundary_conditions,
        constraints=constraints,
    )

    solution = solver.solve()

    return solution


def main() -> None:
    """Run the granular-rves command-line interface."""
    parser = build_parser()
    args = parser.parse_args()

    problem = load_problem(args.problem)

    if problem.analysis == "steady":
        solve_steady(problem)
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


