"""Abstract base class for value functions.

A Value represents a function V(x, t) mapping state-time pairs to scalar values,
typically computed via Hamilton-Jacobi reachability analysis. The value function
encodes safety information: V(x, t) ≥ 0 implies the state is safe.

Implementations should be placed in ``src/impl/values/``.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from .hj_reachability import HJReachabilityDynamics

__all__ = ["Value"]


class Value(abc.ABC):
    """Abstract interface for value functions.

    A value function V(x, t) maps state-time pairs to scalar values representing
    safety or cost. The sign convention is:
        - V(x, t) ≥ 0: state x is safe at time t
        - V(x, t) < 0: state x is unsafe (inside the backward reachable tube)

    Class Attributes:
        hj_dynamics: The HJReachabilityDynamics configuration used to compute this value.
    """

    hj_dynamics: "HJReachabilityDynamics"

    @abc.abstractmethod
    def value(
        self,
        state: torch.Tensor,
        time: Union[torch.Tensor, float],
    ) -> torch.Tensor:
        """Evaluate the value function V(state, time).

        Args:
            state: Tensor of shape ``[..., state_dim]``.
            time: Scalar or tensor of evaluation times.

        Returns:
            Tensor of shape ``[...]`` with value function evaluations.
        """

    @abc.abstractmethod
    def gradient(
        self,
        state: torch.Tensor,
        time: Union[torch.Tensor, float],
    ) -> torch.Tensor:
        """Compute the spatial gradient ∇V(state, time).

        Args:
            state: Tensor of shape ``[..., state_dim]``.
            time: Scalar or tensor of evaluation times.

        Returns:
            Tensor of shape ``[..., state_dim]`` with gradient vectors.
        """
