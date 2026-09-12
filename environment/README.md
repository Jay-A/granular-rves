# Development Environment

`granular-rves` combines Python-based project code with the native scientific-computing stack required by FEniCSx/DOLFINx.

This directory documents the development environment required to build, test, and run the project. The environment description is intentionally kept separate from the Python package dependencies declared in `pyproject.toml`.

## Environment Layers

The project environment consists of several layers:

| Layer | Components | Managed by |
|---|---|---|
| Project | `granular_rves` | `pyproject.toml` |
| Python | Python 3.12 | Python environment |
| Python packages | Gmsh Python interface, pytest, and other Python dependencies | `pyproject.toml` |
| Finite-element framework | FEniCSx / DOLFINx, UFL | Scientific computing environment |
| Numerical infrastructure | PETSc, MPI | Scientific computing environment |
| Native libraries | System libraries required by Gmsh and the FEniCSx stack | Operating system |

The Python virtual environment isolates Python packages, but it does not provide operating-system libraries or the complete native FEniCSx/PETSc/MPI stack.

## Current Development Environment

Development is currently performed on a Linux environment with:

- Python 3.12.14
- Gmsh
- FEniCSx / DOLFINx
- UFL
- PETSc
- MPI
- pytest

The Gmsh Python interface also requires native system libraries. For example, the current Linux development environment requires the library providing `libGLU.so.1`.

These native requirements are part of the project's development environment even though they are not Python package dependencies.

## Python Dependencies

Python-level dependencies are declared in the repository's `pyproject.toml`.

The project uses an editable installation during development:

```bash
python -m pip install -e ".[dev]"
```

This installs the project together with its development dependencies.

## Native Scientific Dependencies

The following components are not currently managed directly by the project Python virtual environment:

- DOLFINx / FEniCSx
- PETSc
- MPI
- operating-system libraries required by Gmsh and the FEniCSx stack

The installation mechanism for these components depends on the scientific-computing environment in which the project is used.

The project therefore does not currently prescribe a single system-level installation method.

## Verification

After the environment has been configured, the project test suite can be run with:

```bash
python -m pytest
```

The meshing integration tests exercise the complete geometry-to-mesh path:

```text
geometry definition
    -> Gmsh geometry
    -> Gmsh mesh
    -> DOLFINx mesh
```

Consequently, these tests require the native scientific-computing environment in addition to the Python dependencies.

## Reproducibility

The environment description will be expanded as the computational stack develops.

In particular, future work may provide a reproducible environment specification for local development and continuous integration. Containerization is intentionally not prescribed at the current stage of the project.
