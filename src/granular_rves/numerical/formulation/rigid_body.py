"""Variational constraints for rigid-body reference motions."""

from __future__ import annotations

from typing import Any

import ufl

from granular_rves.mechanics.rigid_body import RigidBodyConstraintMode


class RigidBodyFormulation:
    """Construct variational constraints for rigid-body reference modes.

    This class translates problem-level rigid-body reference constraints
    into UFL constraint functionals acting on the displacement field.

    The constraints remove arbitrary rigid-body drift without imposing
    pointwise displacement conditions on the reference region.

    Parameters
    ----------
    function_space
        DOLFINx function space containing the displacement field.
    reference_measure
        UFL measure describing the reference region on which the
        constraints are evaluated.
    """

    def __init__(
        self,
        function_space: Any,
        reference_measure: Any,
    ) -> None:
        """Initialize the rigid-body variational formulation."""
        self.function_space = function_space
        self.reference_measure = reference_measure

    def translation_x(self) -> Any:
        """Construct the zero-mean x-translation functional.

        Returns
        -------
        Any
            UFL linear functional

            .. math::

                C_{T_x}(u) =
                \\int_{\\Gamma_\\mathrm{ref}} u_x\\,d\\Gamma.
        """
        displacement = ufl.TrialFunction(self.function_space)
        return displacement[0] * self.reference_measure

    def translation_y(self) -> Any:
        """Construct the zero-mean y-translation functional.

        Returns
        -------
        Any
            UFL linear functional

            .. math::

                C_{T_y}(u) =
                \\int_{\\Gamma_\\mathrm{ref}} u_y\\,d\\Gamma.
        """
        displacement = ufl.TrialFunction(self.function_space)
        return displacement[1] * self.reference_measure

    def translation_z(self) -> Any:
        """Construct the zero-mean z-translation functional.

        Returns
        -------
        Any
            UFL linear functional

            .. math::

                C_{T_z}(u) =
                \\int_{\\Gamma_\\mathrm{ref}} u_z\\,d\\Gamma.
        """
        displacement = ufl.TrialFunction(self.function_space)
        return displacement[2] * self.reference_measure

    def rotation_x(self) -> Any:
        """Construct the zero-mean x-rotation functional.

        Returns
        -------
        Any
            UFL linear functional measuring the rotational component
            about the global x axis.
        """
        displacement = ufl.TrialFunction(self.function_space)
        y = ufl.SpatialCoordinate(self.function_space.mesh)[1]
        z = ufl.SpatialCoordinate(self.function_space.mesh)[2]

        return (y * displacement[2] - z * displacement[1]) * (
            self.reference_measure
        )

    def rotation_y(self) -> Any:
        """Construct the zero-mean y-rotation functional.

        Returns
        -------
        Any
            UFL linear functional measuring the rotational component
            about the global y axis.
        """
        displacement = ufl.TrialFunction(self.function_space)
        x = ufl.SpatialCoordinate(self.function_space.mesh)[0]
        z = ufl.SpatialCoordinate(self.function_space.mesh)[2]

        return (z * displacement[0] - x * displacement[2]) * (
            self.reference_measure
        )

    def rotation_z(self) -> Any:
        """Construct the zero-mean z-rotation functional.

        Returns
        -------
        Any
            UFL linear functional measuring the rotational component
            about the global z axis.
        """
        displacement = ufl.TrialFunction(self.function_space)
        x = ufl.SpatialCoordinate(self.function_space.mesh)[0]
        y = ufl.SpatialCoordinate(self.function_space.mesh)[1]

        return (x * displacement[1] - y * displacement[0]) * (
            self.reference_measure
        )

    def constraints(
        self,
        definition: Any,
    ) -> list[Any]:
        """Construct the active variational rigid-body constraints.

        Parameters
        ----------
        definition
            Problem-level rigid-body constraint definition.

        Returns
        -------
        list[Any]
            UFL linear functionals corresponding to the active
            mean-zero rigid-body constraints.

        Notes
        -----
        Only ``MEAN_ZERO`` constraints are translated here. Physical
        displacement constraints and pointwise rigid-body constraints
        are handled by their respective numerical mechanisms.
        """
        functionals = []

        modes = (
            (definition.translation_x, self.translation_x),
            (definition.translation_y, self.translation_y),
            (definition.translation_z, self.translation_z),
            (definition.rotation_x, self.rotation_x),
            (definition.rotation_y, self.rotation_y),
            (definition.rotation_z, self.rotation_z),
        )

        for mode, functional in modes:
            if mode is RigidBodyConstraintMode.MEAN_ZERO:
                functionals.append(functional())

        return functionals


