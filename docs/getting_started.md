# Getting Started

Granular RVEs is configured through YAML problem-definition files and executed from the command line.

## Installation

Install the project according to the environment described in the repository.

Once the package is available in the active Python environment, verify that the command-line interface is available:

    python -m granular_rves --help

You should see the `granular-rves` command-line interface and its available options.

## Your First Problem

A complete example problem is provided at:

`examples/example.yaml`

It defines a steady small-strain linear-elastic cylinder problem.

The basic execution command is:

    python -m granular_rves examples/example.yaml

For diagnostic output, use:

    python -m granular_rves examples/example.yaml --verbose

The `--verbose` option reports application-level progress while the problem is loaded and executed.

## Problem Definition

A problem-definition file contains the following major sections:

- `name` — problem name.
- `analysis` — analysis type.
- `geometry` — geometric description and its parameters.
- `mesh` — mesh-generation parameters.
- `mechanics` — kinematics, constitutive model, and balance model.
- `boundary` — named problem boundaries and Dirichlet conditions.
- `constraints` — rigid-body constraint configuration.
- `loading` — applied loading.
- `output` — output directory and requested response quantities.

For example:

```yaml
name: example

analysis:
  type: steady

geometry:
  type: cylinder
  x0: [0.0, 0.0, 0.0]
  x1: [0.0, 0.0, 1.0]
  radius: 1.0

mesh:
  size: 0.1
```

The loader validates the required sections and converts the YAML data into structured problem-definition objects before the simulation is started.

## Output

The example configuration writes results to:

    output/example

It also requests reaction-history output and two reaction quantities:

- `top_force` on the `top` boundary in the `z` direction.
- `bottom_force` on the `bottom` boundary in the `z` direction.

See the [Examples](examples) and [User Guide](user_guide) for details on configuring problems.
EOF
```

### `docs/user_guide.md`

```bash
cat > docs/user_guide.md <<'EOF'
# User Guide

Granular RVEs problems are described declaratively using YAML files. The configuration is loaded into structured problem-definition objects and then passed to the simulation runtime.

## Configuration Structure

A problem definition has the following top-level structure:

```yaml
name: example

analysis:
  type: steady

geometry:
  ...

mesh:
  ...

mechanics:
  ...

boundary:
  ...

constraints:
  ...

loading:
  ...

output:
  ...
```

The `constraints` section is optional. The other major sections are required by the problem loader.

## Analysis

The analysis section identifies the analysis type:

```yaml
analysis:
  type: steady
```

The analysis type is stored in the resulting `ProblemDefinition` and determines the downstream simulation path.

## Geometry

Geometry is described using a type and its parameters:

```yaml
geometry:
  type: cylinder
  x0: [0.0, 0.0, 0.0]
  x1: [0.0, 0.0, 1.0]
  radius: 1.0
```

The loader treats the geometry definition declaratively. Geometry construction is performed by downstream geometry components.

For the cylinder example:

- `type` selects the cylinder geometry.
- `x0` defines one endpoint of the cylinder axis.
- `x1` defines the other endpoint.
- `radius` defines the cylinder radius.

## Mesh

Mesh configuration currently specifies the characteristic mesh size:

```yaml
mesh:
  size: 0.1
```

An optional mesh order can also be supplied:

```yaml
mesh:
  size: 0.1
  order: 1
```

If `order` is omitted, the problem loader uses first order.

## Mechanics

The mechanics section selects one model in each of the kinematics, constitutive, and balance categories.

For the example:

```yaml
mechanics:
  kinematics:
    small_strain: {}

  constitutive:
    linear_elastic:
      youngs_modulus: 100000.0
      poisson_ratio: 0.3

  balance:
    momentum: {}
```

Each category must contain exactly one configured model.

### Kinematics

The example uses small-strain kinematics:

```yaml
kinematics:
  small_strain: {}
```

### Constitutive Model

The linear-elastic model is configured with Young's modulus and Poisson's ratio:

```yaml
constitutive:
  linear_elastic:
    youngs_modulus: 100000.0
    poisson_ratio: 0.3
```

### Balance

The momentum balance is selected with:

```yaml
balance:
  momentum: {}
```

## Boundaries

Problem boundaries are named independently of the underlying geometry representation:

```yaml
boundary:
  bottom:
    face: bottom

  top:
    face: top

  lateral:
    face: lateral
```

These names are subsequently used by Dirichlet conditions, rigid-body constraints, loading, and reaction definitions.

The boundary registry therefore provides the common interface between the problem configuration and the numerical boundary implementation.

## Dirichlet Conditions

Dirichlet conditions are defined inside the `boundary` section:

```yaml
boundary:
  bottom:
    face: bottom

  top:
    face: top

  dirichlet:
    bottom:
      component: z
      value: 0.0
```

The referenced boundary must already be declared in the boundary registry.

A Dirichlet definition specifies:

- the named boundary;
- the constrained component;
- the prescribed value.

## Rigid-Body Constraints

Rigid-body constraints are configured separately:

```yaml
constraints:
  reference_boundary: bottom

  rigid_body:
    translation:
      x: mean_zero
      y: mean_zero

    rotation:
      z: mean_zero
```

`reference_boundary` identifies the problem boundary on which the global rigid-body reference functionals are evaluated.

The translation and rotation entries specify the constraint mode for individual rigid-body degrees of freedom.

## Loading

The example applies a displacement to the top boundary:

```yaml
loading:
  type: displacement
  boundary: top
  component: z
  value: -0.05
```

A loading definition identifies:

- `type` — loading type;
- `boundary` — named problem boundary;
- `component` — affected component;
- `value` — loading magnitude.

The referenced boundary must exist in the problem boundary registry.

## Output

Output configuration specifies where results are written:

```yaml
output:
  directory: output/example
```

Reaction-history output can be enabled with:

```yaml
output:
  directory: output/example
  reaction_history: true
```

Individual reactions can then be requested:

```yaml
reactions:
  - name: top_force
    boundary: top
    component: z

  - name: bottom_force
    boundary: bottom
    component: z
```

Each reaction definition has a name, boundary, and component.

## Loading a Problem from Python

The YAML loader is also available directly from Python:

```python
from granular_rves.problem.loader import load_problem

problem = load_problem("examples/example.yaml")
```

A `pathlib.Path` can also be supplied:

```python
from pathlib import Path

from granular_rves.problem.loader import load_problem

problem = load_problem(Path("examples/example.yaml"))
```

`load_problem()` performs configuration parsing and construction of the structured problem definition. It does not perform geometry construction, mesh generation, numerical assembly, or simulation execution.

## Validation

The loader validates the structure of the configuration while it is being loaded.

For example, it checks that:

- required top-level sections are present;
- required fields such as `analysis.type`, `mesh.size`, and `output.directory` are present;
- mechanics sections contain exactly one configured model;
- referenced boundaries have been declared;
- rigid-body constraint modes are supported;
- YAML mappings have the expected structure.

Configuration errors are reported as `ValueError`, while invalid YAML syntax is reported by the YAML parser.

See the [API Reference](api) for the complete Python API.
EOF
```

### `docs/examples.md`

```bash
cat > docs/examples.md <<'EOF'
# Examples

## Steady Elastic Cylinder

The repository contains a complete example problem in:

`examples/example.yaml`

The example defines a three-dimensional cylinder subjected to a prescribed axial displacement.

## Running the Example

From the repository root:

    python -m granular_rves examples/example.yaml

For diagnostic output:

    python -m granular_rves examples/example.yaml --verbose

## Geometry

The cylinder is defined between two points on its axis:

```yaml
geometry:
  type: cylinder
  x0: [0.0, 0.0, 0.0]
  x1: [0.0, 0.0, 1.0]
  radius: 1.0
```

The cylinder therefore has an axial length of `1.0` and a radius of `1.0`.

## Mesh

The example uses:

```yaml
mesh:
  size: 0.1
```

This provides the characteristic mesh size used by the geometry/meshing stage.

## Mechanics

The mechanical model uses small-strain kinematics, linear elasticity, and momentum balance:

```yaml
mechanics:
  kinematics:
    small_strain: {}

  constitutive:
    linear_elastic:
      youngs_modulus: 100000.0
      poisson_ratio: 0.3

  balance:
    momentum: {}
```

The material parameters are:

- Young's modulus: `100000.0`
- Poisson's ratio: `0.3`

## Boundary Conditions

Three named boundaries are declared:

```yaml
boundary:
  bottom:
    face: bottom

  top:
    face: top

  lateral:
    face: lateral
```

The bottom boundary is constrained in the axial direction:

```yaml
dirichlet:
  bottom:
    component: z
    value: 0.0
```

## Rigid-Body Constraints

The example removes additional rigid-body modes using:

```yaml
constraints:
  reference_boundary: bottom

  rigid_body:
    translation:
      x: mean_zero
      y: mean_zero

    rotation:
      z: mean_zero
```

The bottom boundary is used as the reference boundary.

## Applied Loading

The top boundary receives a prescribed negative displacement:

```yaml
loading:
  type: displacement
  boundary: top
  component: z
  value: -0.05
```

Thus the loading acts in the `z` direction with a value of `-0.05`.

## Reaction Output

The example enables reaction-history output:

```yaml
output:
  directory: output/example
  reaction_history: true
```

Two reactions are requested:

```yaml
reactions:
  - name: top_force
    boundary: top
    component: z

  - name: bottom_force
    boundary: bottom
    component: z
```

The resulting output is written below:

    output/example

## Complete Configuration

The complete example is maintained in the repository at:

`examples/example.yaml`

Keeping the executable example in the repository means that the documentation and the actual configuration remain tied to the same input file.

## Related API

The corresponding configuration is loaded by:

`granular_rves.problem.loader.load_problem`

and execution is orchestrated by the simulation runtime.

See the [API Reference](api) for the generated API documentation.
