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
