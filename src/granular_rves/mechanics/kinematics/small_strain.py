from __future__ import annotations

from typing import Any

import ufl


class SmallStrainKinematics:
    """Small-strain kinematics for a displacement field.

    This model computes the infinitesimal strain tensor from a displacement
    field using the symmetric part of its spatial gradient,

    .. math::

        \\boldsymbol{\\varepsilon}(\\mathbf{u})
        =
        \\operatorname{sym}(\\nabla \\mathbf{u}).

    Parameters
    ----------
    parameters
        Model-specific parameters. Small-strain kinematics currently has no
        required parameters, but the mapping is retained for consistency with
        the declarative mechanics definition.

    Notes
    -----
    This class does not create a function space or displacement field.
    The displacement field is supplied to :meth:`strain` by the numerical
    problem layer.
    """

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        """Initialize the small-strain kinematics model."""
        if parameters is None:
            parameters = {}

        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dictionary.")

        if parameters:
            raise ValueError(
                "Small-strain kinematics does not accept parameters."
            )

        self.parameters = parameters

    def strain(self, displacement: Any) -> Any:
        """Compute the infinitesimal strain tensor.

        Parameters
        ----------
        displacement
            Displacement field represented as a UFL expression.

        Returns
        -------
        Any
            Symmetric displacement-gradient tensor represented by UFL.
        """
        return ufl.sym(ufl.grad(displacement))
