from __future__ import annotations

import ufl
from mpi4py import MPI

from granular_rves.mechanics.constitutive.linear_elastic import LinearElastic
from granular_rves.mechanics.kinematics.small_strain import SmallStrainKinematics
from granular_rves.numerical.formulation.elasticity import ElasticityFormulation
from granular_rves.problem.geometry.geometry_types.cylinder import Cylinder
from granular_rves.problem.geometry.meshing import MeshSettings, create_mesh


def make_mesh_data():
    """Create a representative cylinder mesh for formulation tests."""
    geometry = Cylinder(
        x0=(0.0, 0.0, 0.0),
        x1=(0.0, 0.0, 1.0),
        radius=1.0,
    )

    return create_mesh(
        geometry=[geometry],
        settings=MeshSettings(characteristic_length=0.5),
        comm=MPI.COMM_SELF,
    )


def test_elasticity_formulation() -> None:
    """Test construction of the elasticity finite-element formulation."""
    mesh_data = make_mesh_data()

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

    assert formulation.mesh_data is mesh_data
    assert formulation.mesh is mesh_data.mesh
    assert formulation.function_space is not None

    form = formulation.bilinear_form()

    assert isinstance(form, ufl.Form)


