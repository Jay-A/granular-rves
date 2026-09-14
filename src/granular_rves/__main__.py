"""Command-line interface for granular-rves.

This module provides the entry point used by::

    python -m granular_rves <problem.yaml>

The command-line interface is intentionally kept thin. Problem
configuration and simulation orchestration are handled by the
:mod:`granular_rves.problem` package.
"""

from __future__ import annotations

import argparse

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


def main() -> None:
    """Run the granular-rves command-line interface.

    The command-line interface accepts a YAML problem-definition file,
    loads it into a :class:`~granular_rves.problem.definition.ProblemDefinition`,
    and prints a concise summary.

    Simulation execution is delegated to the problem runner.
    """
    parser = build_parser()
    args = parser.parse_args()

    problem = load_problem(args.problem)

    print(
        f"\nProblem: {problem.name}\n"
        f"  Analysis: {problem.analysis.type}\n"
        f"  Geometry: {type(problem.geometry).__name__}\n"
        f"  Mesh size: {problem.mesh.size}\n"
        f"  Loading: {problem.loading.type} ({problem.loading.steps} steps)\n"
        f"  Output: {problem.output.directory}\n"
    )

    print(
       "  Mechanics: \n"
        f"    Kinematics: {problem.mechanics.kinematics.type}\n"
        f"    Constitutive: {problem.mechanics.constitutive.type}\n"
        f"    Balance: {problem.mechanics.balance.type}\n"
    )


if __name__ == "__main__":
    main()
