"""Definitions for rigid-body constraint modes.

This module defines the supported modes for constraining rigid-body
motions in a mechanics problem.
"""

from __future__ import annotations

from enum import Enum


class RigidBodyConstraintMode(str, Enum):
    """Supported constraint modes for rigid-body motions.

    Members
    -------
    UNCONSTRAINED
        Leave the rigid-body motion unconstrained.
    ZERO
        Constrain the rigid-body motion to zero.
    MEAN_ZERO
        Constrain the rigid-body motion using a zero-mean global
        reference condition.
    """

    UNCONSTRAINED = "unconstrained"
    ZERO = "zero"
    MEAN_ZERO = "mean_zero"


class ReferenceFace(str, Enum):
    """Coordinate-aligned reference faces."""

    XM = "xm"
    XP = "xp"
    YM = "ym"
    YP = "yp"
    ZM = "zm"
    ZP = "zp"
