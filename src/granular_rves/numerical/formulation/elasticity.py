"""Finite-element formulation of linear elasticity."""

from __future__ import annotations

from typing import Any

import ufl
from dolfinx import fem


class ElasticityFormulation:
    """Finite-element weak formulation of linear elasticity.

    This class couples the configured mechanics models for kinematics and
    constitutive behaviour to the finite-element formulation used by
    DOLFINx and UFL.

    The steady weak problem is represented as

    .. math::

        a(\\mathbf{u}, \\mathbf{v}) = L(\\mathbf{v}),

    where the bilinear form represents internal virtual work,

    .. math::

        a(\\mathbf{u}, \\mathbf{v})
        =
        \\int_{\\Omega}
        \\boldsymbol{\\sigma}(\\mathbf{u})
        :
        \\boldsymbol{\\varepsilon}(\\mathbf{v})
        \\,\\mathrm{d}\\Omega,

    and the linear form represents external virtual work,

    .. math::

        L(\\mathbf{v})
        =
        \\int_{\\Omega}
        \\mathbf{b}
        \\cdot
        \\mathbf{v}
        \\,\\mathrm{d}\\Omega
        +
        \\int_{\\Gamma_t}
        \\bar{\\mathbf{t}}
        \\cdot
        \\mathbf{v}
        \\,\\mathrm{d}\\Gamma.

    For the current MVP, no body forces or prescribed tractions are
    represented, so ``linear_form`` returns the zero external virtual-work
    form.

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
    variational forms. It does not construct or apply boundary conditions,
    assemble matrices or vectors, or solve the resulting algebraic system.

    Boundary conditions are handled separately by the numerical boundary
    layer. Assembly and numerical solution procedures are handled by the
    solver layer.
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

        For linear elasticity, the resulting form is bilinear in the trial
        and test functions and corresponds to the finite-element stiffness
        operator.
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

    def linear_form(self) -> Any:
        """Construct the elasticity linear form.

        Returns
        -------
        Any
            UFL linear form representing external virtual work.

        Notes
        -----
        The current MVP does not represent body forces or prescribed
        tractions. Consequently, this method returns a zero external
        virtual-work form,

        .. math::

            L(\\mathbf{v})
            =
            \\int_{\\Omega}
            \\mathbf{0}
            \\cdot
            \\mathbf{v}
            \\,\\mathrm{d}\\Omega.

        The form is nevertheless constructed as a genuine UFL linear form
        containing the test function. This allows the solver layer to treat
        the formulation consistently as

        .. math::

            a(\\mathbf{u}, \\mathbf{v}) = L(\\mathbf{v}).

        Prescribed displacements are applied through the numerical boundary
        layer rather than represented in this linear form.
        """
        test_function = ufl.TestFunction(self.function_space)

        zero = ufl.as_vector(
            [ufl.zero() for _ in range(self.mesh.geometry.dim)]
        )

        return ufl.inner(
            zero,
            test_function,
        ) * ufl.dx(domain=self.mesh)


