"""Command-line interface for granular-rves.

This module provides the entry point used by::

    python -m granular_rves <problem.yaml>

The command-line interface owns application-level orchestration and
diagnostic reporting. Numerical algorithms remain in the
:mod:`granular_rves.numerical` package.
"""

from __future__ import annotations

import argparse

from granular_rves.problem.loader import load_problem
from granular_rves.runtime.simulation import SimulationManager


class Reporter:
    """Application-level diagnostic reporter."""

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the reporter.

        Parameters
        ----------
        verbose
            Enable diagnostic output when ``True``.
        """
        self.verbose = verbose

    def __call__(self, message: str) -> None:
        """Report a diagnostic message."""
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

    manager = SimulationManager(
        problem=problem,
        report=report,
    )

    manager.run()


if __name__ == "__main__":
    main()
