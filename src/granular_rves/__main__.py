"""Command-line interface for granular-rves.

This module provides the entry point used by::

    python -m granular_rves <problem.yaml>

The command-line interface is intentionally kept thin. Problem
configuration and simulation orchestration are handled by the
:mod:`granular_rves.problem` package.
"""

from __future__ import annotations

import sys

from granular_rves.problem.loader import load_problem


def main() -> None:
    """Run the granular-rves command-line interface.

    The command-line interface currently accepts a single YAML problem
    definition file. At this stage, the file path is validated only at
    the command-line level and reported to the user. Loading and
    execution of the problem definition will be delegated to the
    problem loader and simulation runner as those components are
    implemented.

    Raises
    ------
    SystemExit
        If the command is invoked without exactly one problem definition
        file.

    Examples
    --------
    Run a problem definition from the command line::

        python -m granular_rves examples/example.yaml
    """
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m granular_rves <problem.yaml>"
        )

    problem = load_problem(sys.argv[1])

    print(
        f" - Problem: {problem.name}\n"
        f"     Analysis: {problem.analysis.type}\n"
        f"     Geometry: {type(problem.geometry).__name__}\n"
        f"     Mesh size: {problem.mesh.size}\n"
        f"     Loading: {problem.loading.type} ({problem.loading.steps} steps)\n"
        f"     Output: {problem.output.directory}"
    )


if __name__ == "__main__":
    main()


