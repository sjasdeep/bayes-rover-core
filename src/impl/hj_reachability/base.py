"""Base class for HJ solver dynamics implementations.

This module provides the mixin that composes the HJReachabilityDynamics interface
with the hj_reachability solver's Dynamics base class.
"""

from __future__ import annotations

from ...core.hj_reachability import HJReachabilityDynamics
from libraries.hj_reachability import hj_reachability as hj  # type: ignore

__all__ = ["HJSolverDynamics"]


class HJSolverDynamics(hj.Dynamics, HJReachabilityDynamics):
    """Base for solver-facing HJ dynamics implementations.

    Combines the JAX-based solver API (hj.Dynamics) with our configuration
    interface (HJReachabilityDynamics). Concrete dynamics inherit from this
    and implement hamiltonian(), partial_max_magnitudes(), etc.
    """

    def __init__(self) -> None:
        # Don't call HJReachabilityDynamics.__init__ - solver dynamics
        # set up channels differently (system comes from class attribute)
        pass

    def validate(self) -> None:  # type: ignore[override]
        HJReachabilityDynamics.validate(self)
