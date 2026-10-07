"""End-to-end test for steady cylinder compression."""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from granular_rves.mechanics.constitutive.linear_elastic import LinearElastic
from granular_rves.mechanics.kinematics.small_strain import SmallStrainKinematics
from granular_rves.numerical.boundary.dirichlet import create_dirichlet_bcs
from granular_rves.numerical.boundary.reference import create_boundary_measure
from granular_rves.numerical.formulation.elasticity import ElasticityFormulation
from granular_rves.numerical.formulation.rigid_body import RigidBodyFormulation
from granular_rves.numerical.solver.steady import SteadySolver
from granular_rves.problem.definition import (
    BoundaryConditionDefinition,
    BoundaryDefinition,
    ConstitutiveDefinition,
    DirichletBoundaryDefinition,
    KinematicsDefinition,
    RigidBodyConstraintDefinition,
)
from granular_rves.problem.geometry.geometry_types.cylinder import Cylinder
from granular_rves.problem.geometry.meshing import (
    MeshSettings,
    create_mesh,
)
from granular_rves.mechanics.rigid_body import RigidBodyConstraintMode


def make_mesh_data():
    """Create the cylinder mesh used by the compression benchmark."""
    geometry = Cylinder(
        x0=(0.0, 0.0, 0.0),
        x1=(0.0, 0.0, 1.0),
        radius=1.0,
    )

    return create_mesh(
        geometry=[geometry],
        settings=MeshSettings(characteristic_length=0.25),
        comm=MPI.COMM_SELF,
    )


def make_boundary_definition() -> BoundaryConditionDefinition:
    """Create the problem boundaries for the compression benchmark."""
    return BoundaryConditionDefinition(
        boundaries={
            "bottom": BoundaryDefinition(face="bottom"),
            "top": BoundaryDefinition(face="top"),
            "lateral": BoundaryDefinition(face="lateral"),
        },
        dirichlet={
            "bottom": DirichletBoundaryDefinition(
                component="z",
                value=0.0,
            ),
            "top": DirichletBoundaryDefinition(
                component="z",
                value=-0.01,
            ),
        },
    )


def make_constraints() -> RigidBodyConstraintDefinition:
    """Create the rigid-body constraints for the compression benchmark."""
    return RigidBodyConstraintDefinition(
        reference_boundary="bottom",
        translation_x=RigidBodyConstraintMode.MEAN_ZERO,
        translation_y=RigidBodyConstraintMode.MEAN_ZERO,
        rotation_z=RigidBodyConstraintMode.MEAN_ZERO,
    )


def test_cylinder_compression() -> None:
    """Solve the steady frictionless cylinder-compression benchmark."""
    mesh_data = make_mesh_data()
    boundary = make_boundary_definition()

    kinematics = SmallStrainKinematics()
    constitutive = LinearElastic(
        youngs_modulus=1.0e6,
        poisson_ratio=0.3,
    )

    formulation = ElasticityFormulation(
        mesh_data=mesh_data,
        kinematics=kinematics,
        constitutive=constitutive,
    )

    boundary_conditions = create_dirichlet_bcs(
        mesh_data=mesh_data,
        function_space=formulation.function_space,
        boundary=boundary,
    )

    reference_measure = create_boundary_measure(
        mesh_data=mesh_data,
        boundary=boundary,
        boundary_name="bottom",
    )

    rigid_body = RigidBodyFormulation(
        function_space=formulation.function_space,
        reference_measure=reference_measure,
    )

    constraints = rigid_body.constraints(make_constraints())

    solver = SteadySolver(
        formulation=formulation,
        boundary_conditions=boundary_conditions,
        constraints=constraints,
    )

    solution = solver.solve()

    assert isinstance(solution, fem.Function)
    assert solution.function_space == formulation.function_space

    displacement = solution.x.array

    assert np.all(np.isfinite(displacement))
    assert not np.allclose(displacement, 0.0)

    bottom_dofs = fem.locate_dofs_topological(
        formulation.function_space.sub(2),
        mesh_data.mesh.topology.dim - 1,
        mesh_data.facet_tags.find(
            mesh_data.physical_groups["bottom"].tag,
        ),
    )

    top_dofs = fem.locate_dofs_topological(
        formulation.function_space.sub(2),
        mesh_data.mesh.topology.dim - 1,
        mesh_data.facet_tags.find(
            mesh_data.physical_groups["top"].tag,
        ),
    )

    top_coordinates = (
        formulation.function_space.sub(2)
        .collapse()[0]
        .tabulate_dof_coordinates()
    )

    assert top_coordinates.shape[0] > 0
    assert top_dofs.size > 0
    assert bottom_dofs.size > 0

    bottom_values = solution.x.array[bottom_dofs]
    top_values = solution.x.array[top_dofs]

    assert np.allclose(
        bottom_values,
        0.0,
        atol=1.0e-10,
    )

    assert np.allclose(
        top_values,
        -0.01,
        atol=1.0e-10,
    )


