from __future__ import annotations

from typing import Any

import ufl


class LinearElastic:
    """Isotropic linear-elastic constitutive model.

    The model relates the small-strain tensor to the Cauchy stress tensor
    according to

    .. math::

        \\boldsymbol{\\sigma}
        =
        \\lambda \\operatorname{tr}(\\boldsymbol{\\varepsilon})\\mathbf{I}
        +
        2\\mu\\boldsymbol{\\varepsilon}.

    Parameters
    ----------
    youngs_modulus
        Young's modulus of the material. Must be positive.
    poisson_ratio
        Poisson's ratio of the material. Must satisfy
        ``-1 < poisson_ratio < 0.5``.

    Raises
    ------
    TypeError
        If either material parameter is not a real number.
    ValueError
        If the material parameters are outside their physically admissible
        ranges.

    Notes
    -----
    The displacement field and strain tensor are supplied by other parts of
    the mechanics framework. This class does not create function spaces,
    displacement fields, meshes, or variational problems.
    """

    def __init__(
        self,
        youngs_modulus: float,
        poisson_ratio: float,
    ) -> None:
        """Initialize the linear-elastic material model."""
        if not isinstance(youngs_modulus, (int, float)):
            raise TypeError("youngs_modulus must be a real number.")

        if not isinstance(poisson_ratio, (int, float)):
            raise TypeError("poisson_ratio must be a real number.")

        if youngs_modulus <= 0.0:
            raise ValueError("youngs_modulus must be positive.")

        if not -1.0 < poisson_ratio < 0.5:
            raise ValueError(
                "poisson_ratio must satisfy -1 < poisson_ratio < 0.5."
            )

        self.youngs_modulus = float(youngs_modulus)
        self.poisson_ratio = float(poisson_ratio)

        self.shear_modulus = (
            self.youngs_modulus
            / (2.0 * (1.0 + self.poisson_ratio))
        )

        self.lame_first_parameter = (
            self.youngs_modulus
            * self.poisson_ratio
            / (
                (1.0 + self.poisson_ratio)
                * (1.0 - 2.0 * self.poisson_ratio)
            )
        )

    def stress(self, strain: Any) -> Any:
        """Compute the Cauchy stress from the small-strain tensor.

        Parameters
        ----------
        strain
            Small-strain tensor represented as a UFL expression.

        Returns
        -------
        Any
            Cauchy stress tensor represented as a UFL expression.
        """
        identity = ufl.Identity(strain.ufl_shape[0])

        return (
            self.lame_first_parameter * ufl.tr(strain) * identity
            + 2.0 * self.shear_modulus * strain
        )
