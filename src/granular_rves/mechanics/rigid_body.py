"""Definitions for rigid-body constraint modes.

This module defines the mechanics-level vocabulary used to describe
constraints on rigid-body motions. It contains no numerical, mesh, or
boundary-resolution logic.
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
