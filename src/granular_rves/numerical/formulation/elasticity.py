from __future__ import annotations

from typing import Any

import ufl
from dolfinx import fem


class ElasticityFormulation:
    """Finite-element formulation of an elasticity problem.

    This class couples the mechanics models for kinematics and constitutive
    behaviour to the finite-element formulation used by DOLFINx and UFL.

    The formulation currently represents the internal virtual work,

    .. math::

        a(\\mathbf{u}, \\mathbf{v})
        =
        \\int_{\\Omega}
        \\boldsymbol{\\sigma}(\\mathbf{u})
        :
        \\boldsymbol{\\varepsilon}(\\mathbf{v})
        \\,\\mathrm{d}\\Omega,

    where the strain and stress relations are supplied by the configured
    mechanics models.

    Parameters
    ----------
    mesh_data
        DOLFINx ``MeshData`` returned by the mesh-generation layer.
        The object must provide a ``mesh`` attribute containing the
        computational DOLFINx mesh.
    kinematics
        Mechanics kinematics model providing the strain relation.
    constitutive
        Mechanics constitutive model providing the stress relation.

    Notes
    -----
    This class constructs the finite-element function space and UFL
    variational forms, but does not construct boundary conditions, assemble
    the system, or solve it.

    Boundary conditions are handled separately by the numerical boundary
    layer. Numerical solution procedures are handled by the solver layer.
    """

    def __init__(
        self,
        mesh_data: Any,
        kinematics: Any,
        constitutive: Any,
    ) -> None:
        """Initialize the elasticity finite-element formulation.

        Parameters
        ----------
        mesh_data
            DOLFINx ``MeshData`` returned by the mesh-generation layer.
        kinematics
            Mechanics kinematics model providing the strain relation.
        constitutive
            Mechanics constitutive model providing the stress relation.
        """
        self.mesh_data = mesh_data
        self.mesh = mesh_data.mesh
        self.kinematics = kinematics
        self.constitutive = constitutive

        self.function_space = fem.functionspace(
            self.mesh,
            ("Lagrange", 1, (self.mesh.geometry.dim,)),
        )

    def bilinear_form(self) -> Any:
        """Construct the elasticity bilinear form.

        Returns
        -------
        Any
            UFL bilinear form representing the internal virtual work.

        Notes
        -----
        The displacement trial function is passed through the configured
        kinematics and constitutive models to construct the stress response.
        The test function is passed through the kinematics model to obtain
        the virtual strain.
        """
        displacement = ufl.TrialFunction(self.function_space)
        test_function = ufl.TestFunction(self.function_space)

        strain = self.kinematics.strain(displacement)
        stress = self.constitutive.stress(strain)
        virtual_strain = self.kinematics.strain(test_function)

        return ufl.inner(
            stress,
            virtual_strain,
        ) * ufl.dx(domain=self.mesh)


