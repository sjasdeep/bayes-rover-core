"""Abstract base class for system inputs.

An Input represents a signal (control, disturbance, or uncertainty) that can be
evaluated at any (state, time) pair. Inputs can be standalone (e.g., MPC controllers)
or derived from cached data (e.g., GridInput, NNInput).

Implementations should be placed in ``src/impl/inputs/``.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Literal, Type

import torch

if TYPE_CHECKING:
    from .systems import System

__all__ = ["Input", "InputType"]

InputType = Literal["any", "control", "disturbance", "uncertainty"]


class Input(abc.ABC):
    """Abstract interface for system input signals.

    An Input is a state-feedback policy that maps (state, time) to an input vector.
    The same interface is used for controls, disturbances, and uncertainty signals.

    Class Attributes:
        type: Channel type - one of 'any', 'control', 'disturbance', 'uncertainty'.
            Use 'any' for generic inputs (e.g., ZeroInput) that work with any channel.
        system_class: The System class this input is compatible with.
        dim: Dimension of the input vector.
        time_invariant: If True, input does not depend on time.
    """

    type: InputType
    system_class: Type["System"]
    dim: int
    time_invariant: bool = False

    def set_type(self, input_type: InputType) -> None:
        """Set the channel type for this input.

        This is used for generic inputs (type='any') that can serve as
        control, disturbance, or uncertainty signals.

        Args:
            input_type: The channel type to assign.

        Raises:
            ValueError: If input_type is invalid or conflicts with existing type.
        """
        if input_type not in {"any", "control", "disturbance", "uncertainty"}:
            raise ValueError(f"Invalid input type: {input_type}")
        if self.type != "any" and self.type != input_type:
            raise ValueError(
                f"Input type already set to {self.type}, cannot change to {input_type}"
            )
        self.type = input_type

    @abc.abstractmethod
    def input(
        self,
        state: torch.Tensor,
        time: float,
    ) -> torch.Tensor:
        """Evaluate the input signal at (state, time).

        Args:
            state: Tensor of shape ``[..., state_dim]``.
            time: Scalar simulation time in seconds.

        Returns:
            Tensor of shape ``[..., dim]`` with the input values.
        """

    @abc.abstractmethod
    def bind(self, system: "System") -> None:
        """Bind this input to a specific System instance.

        This method is called before the input is used, allowing it to
        configure itself based on the system's properties (e.g., dimensions,
        limits, or other parameters).

        Args:
            system: The System instance to bind to.
        """

    def reset(self) -> None:
        """Reset any internal state before a new simulation.

        Override this method in inputs that maintain internal state.
        Called automatically at the start of each simulation.

        Default implementation does nothing.
        """
        pass
