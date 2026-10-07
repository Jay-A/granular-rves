"""Utilities for writing simulation results.

This module provides the minimal output path required to write solved
finite-element fields for visualization in ParaView.
"""

from __future__ import annotations

from pathlib import Path

from dolfinx.io import VTKFile


def write_solution(
    mesh_data,
    solution,
    output_directory: str | Path,
) -> None:
    """Write a solved displacement field for visualization.

    Parameters
    ----------
    mesh_data
        DOLFINx mesh data associated with the solved problem.
    solution
        Computed displacement field.
    output_directory
        Directory in which the visualization output is written.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = output_directory / "solution.pvd"

    with VTKFile(
        mesh_data.mesh.comm,
        str(output_file),
        "w",
    ) as vtk:
        vtk.write_mesh(mesh_data.mesh)
        vtk.write_function(solution)




def write_reaction_history(
    reaction_history,
    output_directory,
):
    """Write reaction history to CSV."""

    import csv

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = output_directory / "reaction_history.csv"

    with output_file.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["name", "step", "time", "reaction"],
        )
        writer.writeheader()
        writer.writerows(reaction_history)

    return output_file
