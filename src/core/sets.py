"""Abstract base class for state-conditioned input sets.

A Set represents the collection of possible input values at each (state, time) pair.
This is used to capture the effect of state uncertainty on control inputs: when the
controller observes a noisy state estimate, the resulting control lies within a set
rather than being a single value.

Implementations should be placed in ``src/impl/sets/``.
"""

from __future__ import annotations

import abc
from typing import Literal, Tuple

import torch

__all__ = ["Set", "SetType"]

SetType = Literal["box", "hull"]


class Set(abc.ABC):
    """Abstract interface for state-conditioned input sets.

    A Set maps (state, time) to a set of possible input values. This is used
    in robust verification to capture uncertainty-induced variability in control.

    Class Attributes:
        set_type: Representation type - 'box' (axis-aligned bounds) or 'hull' (convex hull).
        dim: Dimension of the input space.
        time_invariant: If True, the set does not depend on time.
    """

    set_type: SetType
    dim: int
    time_invariant: bool = False

    @abc.abstractmethod
    def as_box(
        self,
        state: torch.Tensor,
        time: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return axis-aligned bounding box of the set at (state, time).

        Args:
            state: Tensor of shape ``[..., state_dim]``.
            time: Scalar simulation time.

        Returns:
            Tuple ``(lower, upper)`` each of shape ``[..., dim]``.
        """

    @abc.abstractmethod
    def argmax_support(
        self,
        direction: torch.Tensor,
        state: torch.Tensor,
        time: float,
    ) -> torch.Tensor:
        """Find the point in the set that maximizes the dot product with direction.

        This implements the support function argmax, returning the element u* such that
        ``<direction, u*> = max_{u in Set} <direction, u>``.

        Args:
            direction: Tensor of shape ``[..., dim]``.
            state: Tensor of shape ``[..., state_dim]``.
            time: Scalar simulation time.

        Returns:
            Tensor of shape ``[..., dim]`` with the extremal elements.
        """
