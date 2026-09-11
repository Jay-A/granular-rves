# granular-rves

Three-dimensional computational mechanics of realistic granular microstructures.

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jay-a.github.io/granular-rves/)

## Technologies

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![FEniCSx](https://img.shields.io/badge/FEniCSx-3D%20FEM-blue)
![DOLFINx](https://img.shields.io/badge/DOLFINx-FEM-blue)
![PETSc](https://img.shields.io/badge/PETSc-Solvers-00599C)
![UFL](https://img.shields.io/badge/UFL-Variational%20Forms-4B8BBE)
![Sphinx](https://img.shields.io/badge/Sphinx-Documentation-000000?logo=sphinx&logoColor=white)
![MyST](https://img.shields.io/badge/MyST-Markdown-526CFE)
![PyData](https://img.shields.io/badge/PyData%20Sphinx%20Theme-Documentation-F37626)
![LaTeX](https://img.shields.io/badge/LaTeX-Report-008080?logo=latex&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI-2088FF?logo=githubactions&logoColor=white)
![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Documentation-222222?logo=githubpages&logoColor=white)

---

**Status:** Early development

## Overview

`granular-rves` is a dedicated [FEniCSx](https://github.com/FEniCS) application for three-dimensional computational mechanics of realistic granular microstructures.

The project develops finite-element models of individual grains and granular assemblies with an emphasis on:

* realistic three-dimensional grain geometry
* nonlinear solid mechanics
* grain-to-grain contact
* periodic representative volume elements (RVEs)
* homogenization and effective response
* model calibration and parameter identification
* uncertainty quantification
* data-driven surrogate modeling
* eventual integration with scientific machine learning methods

The immediate goal is to establish a verified computational mechanics workflow from a single deformable grain to periodic granular RVEs. Longer-term work will investigate how high-fidelity finite-element simulations can be combined with reduced-order and operator-learning approaches.

This repository is intentionally science-first. The computational models, numerical methods, geometry, verification, and reproducibility are developed before introducing machine-learning surrogates.

## Scientific scope

The long-term objective is to construct a computational map of the form

```text
{material parameters, grain geometry, microstructure, loading}
                            |
                            v
                    finite-element model
                            |
                            v
                     stress-strain response
```

A representative formulation is

```text
G:
{E, nu, mu, rho, microstructure, loading}
    -> sigma(epsilon)
```

where the input describes material properties, grain geometry and arrangement, and loading conditions, while the output is the resulting mechanical response.

The eventual research direction is to learn useful approximations of this map for parameterized granular systems while retaining a clear connection to the underlying mechanics.

## Computational workflow

The project progresses from controlled finite-element problems toward realistic granular microstructures and, ultimately, data-driven surrogate models.

| Stage | Computational problem | Primary capability                                      |
| ----- | --------------------- | ------------------------------------------------------- |
| 1     | Single grain          | Three-dimensional mesh loading and linear elasticity    |
| 2     | Body force            | Distributed loading and equilibrium                     |
| 3     | Nonlinear grain       | Hyperelastic constitutive modeling                      |
| 4     | Two grains            | Deformable grain contact                                |
| 5     | Periodic stack        | Periodic boundary conditions                            |
| 6     | Periodic packing      | Three-dimensional granular microstructure               |
| 7     | Realistic grains      | Experimentally informed grain geometry                  |
| 8     | Homogenization        | Macroscopic response and effective properties           |
| 9     | Calibration and UQ    | Parameter identification and uncertainty quantification |
| 10    | Surrogate modeling    | Reduced-order and data-driven prediction                |
| 11    | Scientific ML         | PINNs and neural operators                              |

Each stage introduces additional physical or numerical complexity while preserving a path toward verification.

## Current development

The repository is in early development.

The initial implementation focuses on establishing the computational infrastructure needed for reliable three-dimensional finite-element simulations in FEniCSx.

The first stages target:

1. loading and inspecting a three-dimensional grain surface mesh
2. generating a finite-element volume mesh
3. solving single-grain linear elasticity
4. applying body forces
5. introducing nonlinear constitutive behavior
6. introducing contact between deformable grains
7. constructing periodic granular configurations

More advanced capabilities will be added only after the underlying mechanics and numerical implementation have been tested.

## Configuration-driven experiments

Physical experiments are described using YAML configuration files rather than embedding experiment definitions directly in Python code.

For example:

```yaml
name: single_grain_linear_elastic

geometry:
  type: mesh
  path: meshes/examples/quartz_grain.stl

material:
  model: linear_elastic
  youngs_modulus: 70.0e9
  poissons_ratio: 0.20

loading:
  type: uniaxial_compression
  direction: z
  displacement: -1.0e-6

solver:
  nonlinear: false

output:
  format: vtu
  directory: results/01_single_grain
```

The YAML file describes the physical experiment and high-level modeling choices.

The Python package is responsible for implementing:

* geometry handling
* mesh generation
* material models
* constitutive equations
* finite-element formulations
* contact
* boundary conditions
* loading
* numerical solution procedures
* output
* post-processing

This separation is intended to make computational experiments reproducible while keeping implementation details inside the package.

## Installation

The project uses `uv` for Python environment and package management.

Development is currently performed with Python 3.12.14. Python compatibility requirements will remain intentionally lightweight while the dependency tree is developing and will be refined as the FEniCSx, PETSc, mesh-generation, visualization, and scientific Python dependencies become established.

Clone the repository and synchronize the development environment:

```bash
git clone https://github.com/jay-a/granular-rves.git
cd granular-rves

uv sync
```

Once the application entry point is available, experiments can be run through the managed environment:

```bash
uv run granular-rves run experiments/01_single_grain.yaml
```

`pyproject.toml` is the source of truth for project metadata and dependencies. `uv` manages the project environment and lockfile.

## Software stack

The project is being developed around the following ecosystem:

* Python
* FEniCSx
* DOLFINx
* UFL
* PETSc
* Gmsh
* NumPy
* SciPy
* PyVista
* pytest
* uv
* Sphinx
* MyST
* PyData Sphinx Theme
* LaTeX

The exact dependency tree will evolve as the implementation grows.

## FEniCSx and DOLFINx

This project is a dedicated application built on the FEniCSx finite-element ecosystem.

[FEniCSx](https://github.com/FEniCS) provides the broader finite-element problem-solving environment. [DOLFINx](https://github.com/FEniCS/dolfinx) provides the computational environment used to implement finite-element problems in Python and C++.

The project is particularly interested in using FEniCSx as a programmable framework for developing mechanics applications rather than treating finite-element analysis as a black-box workflow.

The implementation will therefore emphasize:

* explicit constitutive models
* explicit weak formulations
* programmable finite-element spaces
* boundary-condition construction
* numerical solution procedures
* verification tests
* reproducible computational experiments

## Documentation

Project documentation is built with:

* Sphinx
* MyST
* PyData Sphinx Theme
* Sphinx bibliography support

[**Read the documentation**](https://jay-a.github.io/granular-rves/)

The documentation will contain:

* getting-started material
* theoretical background
* computational examples
* numerical methods
* verification studies
* API documentation
* research notes
* references
* generated technical reports

> Documentation tooling: The project uses Sphinx with MyST and the PyData Sphinx Theme. This stack was chosen in part based on the documentation tooling used by DOLFINx, the FEniCSx finite-element framework this project builds upon. See the [DOLFINx Python package configuration](https://github.com/FEniCS/dolfinx/blob/main/python/pyproject.toml) for its documentation dependencies.

The HTML documentation and technical PDF report will be generated automatically through GitHub Actions and published through GitHub Pages.

## Verification and validation

Verification is treated as a core part of the project rather than an afterthought.

Early verification problems will include:

* analytical solutions for linear elasticity
* convergence with mesh refinement
* manufactured solutions where appropriate
* symmetry checks
* equilibrium checks
* energy consistency
* sensitivity to solver tolerances
* contact benchmark problems
* periodic-boundary consistency

As the models become more realistic, validation will compare selected computational results against experimental or published data where appropriate.

The distinction between verification and validation will be maintained throughout the project.

## Reproducibility

The project is intended to make computational experiments reproducible from configuration files and version-controlled code.

A typical experiment should eventually be reproducible with a command of the form:

```bash
uv run granular-rves run experiments/01_single_grain.yaml
```

Experiment configuration files will define the physical problem, while the package provides the implementation.

Generated simulation results are not intended to be committed to the repository by default. Instead, the repository will contain:

* input configurations
* source code
* meshes or links to externally hosted geometry
* tests
* documentation
* scripts needed to reproduce results
* selected lightweight reference outputs where useful

Large generated datasets and simulation outputs should be handled separately when appropriate.

## Project structure

The repository is organized around the scientific workflow rather than around a single executable script.

```text
granular-rves/
+-- README.md
+-- LICENSE
+-- pyproject.toml
+-- .gitignore
+-- .pre-commit-config.yaml
|
+-- .github/
|   +-- workflows/
|       +-- docs.yml
|
+-- docs/
|   +-- conf.py
|   +-- index.md
|   +-- getting_started/
|   +-- theory/
|   +-- examples/
|   +-- api/
|   +-- report/
|       +-- main.tex
|       +-- references.bib
|       +-- figures/
|
+-- src/
|   +-- granular_rves/
|       +-- geometry/
|       +-- mesh/
|       +-- materials/
|       +-- mechanics/
|       +-- contact/
|       +-- boundary/
|       +-- loading/
|       +-- experiments/
|       +-- io/
|       +-- postprocess/
|
+-- experiments/
|   +-- 01_single_grain.yaml
|   +-- 02_body_force.yaml
|   +-- 03_periodic_stack.yaml
|   +-- 04_periodic_packing.yaml
|
+-- examples/
|   +-- 01_single_grain/
|   +-- 02_body_force/
|   +-- 03_periodic_stack/
|   +-- 04_periodic_packing/
|
+-- meshes/
|   +-- README.md
|   +-- examples/
|
+-- tests/
|   +-- test_materials.py
|   +-- test_mesh.py
|   +-- test_elasticity.py
|
+-- results/
    +-- .gitkeep
```

The structure will evolve as the implementation grows.

## Geometry

Realistic grain geometry is a central component of the project.

The long-term objective is to move beyond idealized spheres and simple synthetic shapes toward experimentally informed three-dimensional grain geometries.

Candidate sources include:

* synchrotron micro-computed tomography
* experimentally reconstructed sand grains
* published grain geometry datasets
* synthetic grains generated from experimentally observed morphology
* project-specific grain-generation workflows

The initial materials focus on quartz and silica sand because these provide a useful connection between realistic grain morphology, experimentally observed granular behavior, and computational mechanics.

## Grain geometry references and data sources

The project draws on published work in three-dimensional characterization and generation of realistic sand grains.

### Vlassis et al. realistic sand grain data

Vlassis, Sun, Alshibli, and Regueiro developed a denoising-diffusion approach for generating realistic sand grains in latent space from three-dimensional F50 sand grain data.

The work uses an experimentally derived database of three-dimensional sand grain geometries and also reports the generation and release of synthetic grain geometries.

Reference:

> Vlassis, N. N., Sun, W., Alshibli, K. A., and Regueiro, R. A. (2024). "Synthesizing realistic sand assemblies with denoising diffusion in latent space." *International Journal for Numerical and Analytical Methods in Geomechanics*, 48(16), 3933-3956. DOI: 10.1002/nag.3818.

* Paper: https://doi.org/10.1002/nag.3818
* Mendeley Data: https://www.mendeley.com/catalogue/3b5b22c4-b4f2-3e11-9df2-dae20013749f/
* Preprint: https://arxiv.org/abs/2306.04411

The Vlassis et al. work is particularly relevant to this project because it provides experimentally grounded grain geometry together with a pathway toward synthetic morphology generation.

### Alshibli and collaborators

The project also draws on the extensive three-dimensional sand morphology and particle-scale mechanics work of Khalid Alshibli and collaborators.

In particular, Alshibli's research documents the use of synchrotron micro-computed tomography to reconstruct realistic three-dimensional sand particles for computational mechanics. These workflows provide experimentally derived particle surfaces that can be used in numerical models.

Relevant resources include:

* Alshibli research group: https://alshibli.utk.edu/research/
* 3D synchrotron micro-computed tomography: https://alshibli.utk.edu/research/3d-synchrotron-micro-computed-tomography-smt/
* Silica sand particle fracture and FE modeling: https://alshibli.utk.edu/research/stresses-within-particles-force-chains-and-fracture-behavior-of-silica-sand-in-3d/
* Publications: https://alshibli.utk.edu/publications/

Relevant publications include:

> Druckrey, A. M., Alshibli, K., and Al-Raoush, R. (2016). "3D Characterization of Sand Particle-to-Particle Contact and Morphology." *Computers and Geotechnics*, 74, 26-35. DOI: 10.1016/j.compgeo.2015.12.014.

> Druckrey, A. M. and Alshibli, K. A. (2016). "3D Finite Element Modeling of Sand Particle Fracture based on in situ X-Ray Synchrotron Imaging." *International Journal for Numerical and Analytical Methods in Geomechanics*, 40, 105-116. DOI: 10.1002/nag.2396.

> Alshibli, K. A., Druckrey, A. M., Al-Raoush, R., Weiskittel, T., and Lavrik, N. V. (2015). "Quantifying morphology of sands using 3D imaging." *Journal of Materials in Civil Engineering*, 27(10).

These sources provide scientific context for the use of realistic particle morphology in finite-element and particle-scale granular mechanics.

## Research direction

The broader research direction is motivated by the computational cost of repeatedly solving detailed finite-element models of complex microstructures.

| Model input         | High-fidelity computation      | Model output           |
| ------------------- | ------------------------------ | ---------------------- |
| Material parameters | FEniCSx finite-element solve   | Stress-strain response |
| Grain morphology    | Constitutive evaluation        | Contact response       |
| Grain arrangement   | Nonlinear solution             | Local fields           |
| Contact state       | Boundary-condition enforcement | Macroscopic response   |
| Loading history     | Numerical solution             | Effective properties   |

For parameter studies, inverse problems, uncertainty quantification, and real-time applications, repeatedly evaluating this model can become expensive.

The long-term objective is to investigate whether this high-fidelity computational map can be approximated by a data-driven model while retaining:

* physical interpretability
* numerical verification
* uncertainty awareness
* parameter sensitivity
* reproducibility
* a clear connection to the high-fidelity model

This creates a natural progression from classical computational mechanics to scientific machine learning.

## Roadmap

### Phase 1: Finite-element foundations

* [ ] Project packaging and development environment
* [ ] FEniCSx/DOLFINx application structure
* [ ] Mesh import and inspection
* [ ] Single-grain linear elasticity
* [ ] Analytical verification
* [ ] Mesh convergence studies
* [ ] Body-force loading
* [ ] VTK-compatible output and visualization

### Phase 2: Nonlinear mechanics

* [ ] Hyperelastic material formulation
* [ ] Neo-Hookean single-grain model
* [ ] Nonlinear solver configuration
* [ ] Large-deformation verification
* [ ] Consistent post-processing of stresses and strains

### Phase 3: Grain contact

* [ ] Two-grain geometry
* [ ] Frictionless contact
* [ ] Contact verification
* [ ] Small-to-moderate deformation contact benchmarks
* [ ] Extension toward frictional contact

### Phase 4: Periodic granular microstructures

* [ ] Periodic grain stack
* [ ] Triply periodic boundary conditions
* [ ] Periodic granular packing
* [ ] Realistic grain geometries
* [ ] Macroscopic stress and strain measures
* [ ] Representative volume element studies

### Phase 5: Homogenization and calibration

* [ ] Effective material response
* [ ] Homogenized constitutive quantities
* [ ] Parameter identification
* [ ] Bayesian calibration
* [ ] Uncertainty quantification
* [ ] Experimental-data integration

### Phase 6: Scientific machine learning

* [ ] Reduced-order representations
* [ ] Surrogate models
* [ ] Physics-informed neural networks
* [ ] DeepONet
* [ ] Fourier neural operators
* [ ] Parameterized operator learning
* [ ] Uncertainty-aware surrogate prediction
* [ ] Comparison against high-fidelity FEniCSx simulations

The machine-learning stages are intentionally downstream of the finite-element and verification work. The objective is to build surrogates whose behavior can be evaluated against a well-defined high-fidelity computational model.

## Development philosophy

The project follows several principles.

### Mechanics before machine learning

Machine-learning models should be built on top of a clearly defined and verified computational mechanics problem.

### Verification before acceleration

A fast surrogate is only useful if the reference solution is trustworthy.

### Reproducibility by configuration

Scientific experiments should be defined by version-controlled inputs rather than hidden implementation details.

### Explicit numerical methods

Constitutive equations, boundary conditions, discretizations, and solver choices should remain inspectable.

### Modular scientific software

Geometry, materials, mechanics, contact, loading, solvers, I/O, and post-processing should remain separable components.

### Honest scope

Features are documented as implemented, experimental, or planned. The roadmap should not be confused with current capability.

## Contributing

The project is currently in early development, and before making contributions please reach out, but.
as the implementation stabilizes, contributions may include:

* verification benchmarks
* constitutive models
* contact formulations
* mesh-generation workflows
* realistic grain geometries
* homogenization methods
* uncertainty quantification
* surrogate-model implementations
* documentation
* tests and reproducibility improvements

Development practices will emphasize automated testing, code quality, documentation, and reproducible numerical experiments.

## Citation

If this software contributes to published research, please cite the repository using the project's `CITATION.cff` file once available.

The scientific sources that motivate the realistic grain geometry component should also be cited where their data, methods, or scientific results are used.

## Author

Jay M. Appleton

GitHub: [jay-a](https://github.com/jay-a)

## License

This project is released under the MIT License.

See [LICENSE](LICENSE) for the full license text.

## Acknowledgments

This project builds on the open-source FEniCSx ecosystem and on published experimental and computational work in granular materials, three-dimensional particle morphology, synchrotron micro-computed tomography, and scientific machine learning.

Particular scientific context is provided by the work of Nikolaos Vlassis, WaiChing Sun, Khalid Alshibli, Richard Regueiro, and their collaborators on realistic sand grain generation and particle-scale granular mechanics.

## Status

This repository is under active development.

The current priority is to establish a robust, tested, and reproducible FEniCSx computational mechanics workflow for realistic three-dimensional granular microstructures before expanding toward homogenization, uncertainty quantification, and scientific machine learning.

