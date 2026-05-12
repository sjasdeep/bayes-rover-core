"""Abstract base class for dynamical systems.

A System defines the continuous-time dynamics, constraints, and visualization
for a controlled dynamical system with disturbances and state uncertainty.

Implementations should be placed in ``src/impl/systems/``.
"""

from __future__ import annotations

import abc
from typing import List, Tuple

import torch
from matplotlib.axes import Axes

__all__ = ["System"]


class System(abc.ABC):
    """Abstract interface for dynamical systems.

    A System encapsulates:
        - State-space structure (dimension, limits, periodicity)
        - Control and disturbance input spaces
        - Continuous-time dynamics f(x, u, d, t)
        - Constraint sets (failure/unsafe regions, goal regions)
        - State uncertainty bounds (for robust verification)

    Class Attributes:
        state_dim: Number of state dimensions.
        state_limits: Tensor of shape ``[2, state_dim]`` with [lower, upper] bounds.
        state_periodic: List indicating which state dimensions wrap around.
        state_labels: Display labels for each state dimension.
        control_dim: Number of control dimensions.
        control_labels: Display labels for each control dimension.
        disturbance_dim: Number of disturbance dimensions.
        disturbance_labels: Display labels for each disturbance dimension.
        initial_state: Default initial state tensor of shape ``[state_dim]``.
        time_invariant_uncertainty_limits: If True, uncertainty bounds don't vary with time.
        time_horizon: Default time horizon for simulations/reachability.
    """

    state_dim: int
    state_limits: torch.Tensor  # Shape: [2, state_dim]
    state_periodic: List[bool]
    state_labels: Tuple[str, ...]

    control_dim: int
    control_labels: Tuple[str, ...]

    disturbance_dim: int
    disturbance_labels: Tuple[str, ...]

    initial_state: torch.Tensor

    time_invariant_uncertainty_limits: bool = False

    time_horizon: float

    @abc.abstractmethod
    def control_limits(
        self,
        state: torch.Tensor,
        time: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return admissible control bounds at (state, time).

        Args:
            state: Tensor of shape ``[..., state_dim]`` (batchable).
            time: Scalar simulation time.

        Returns:
            Tuple ``(lower, upper)`` each of shape ``[..., control_dim]``.
        """

    @abc.abstractmethod
    def disturbance_limits(
        self,
        state: torch.Tensor,
        time: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return disturbance bounds at (state, time).

        Args:
            state: Tensor of shape ``[..., state_dim]``.
            time: Scalar simulation time.

        Returns:
            Tuple ``(lower, upper)`` each of shape ``[..., disturbance_dim]``.
        """

    @abc.abstractmethod
    def uncertainty_limits(
        self,
        state: torch.Tensor,
        time: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return bounds on state-estimation error at (state, time).

        The uncertainty ``e`` represents the difference between the true state
        and the estimated state: ``x_estimated = x_true + e``.

        Args:
            state: Tensor of shape ``[..., state_dim]``.
            time: Scalar simulation time.

        Returns:
            Tuple ``(lower, upper)`` each of shape ``[..., state_dim]``.
        """

    @abc.abstractmethod
    def failure_function(
        self,
        state: torch.Tensor,
        time: float,
    ) -> torch.Tensor:
        """Signed distance to the failure set.

        Negative values indicate unsafe states (inside the failure set).

        Args:
            state: Tensor of shape ``[..., state_dim]``.
            time: Scalar simulation time.

        Returns:
            Tensor of shape ``[...]`` with signed distances.
        """

    @abc.abstractmethod
    def goal_function(
        self,
        state: torch.Tensor,
        time: float,
    ) -> torch.Tensor:
        """Signed distance to the goal set.

        Negative values indicate the state is inside the goal set.

        Args:
            state: Tensor of shape ``[..., state_dim]``.
            time: Scalar simulation time.

        Returns:
            Tensor of shape ``[...]`` with signed distances.
        """

    @abc.abstractmethod
    def dynamics(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        disturbance: torch.Tensor,
        time: float,
    ) -> torch.Tensor:
        """Continuous-time dynamics dx/dt = f(x, u, d, t).

        Args:
            state: Tensor of shape ``[..., state_dim]``.
            control: Tensor of shape ``[..., control_dim]``.
            disturbance: Tensor of shape ``[..., disturbance_dim]``.
            time: Scalar simulation time.

        Returns:
            Tensor of shape ``[..., state_dim]`` with time derivatives.
        """

    @abc.abstractmethod
    def render(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        disturbance: torch.Tensor,
        uncertainty: torch.Tensor,
        time: float,
        ax: Axes,
    ) -> None:
        """Render the system state on a matplotlib axes.

        Args:
            state: Tensor of shape ``[..., state_dim]``.
            control: Tensor of shape ``[..., control_dim]``.
            disturbance: Tensor of shape ``[..., disturbance_dim]``.
            uncertainty: Tensor of shape ``[..., state_dim]``.
            time: Scalar simulation time.
            ax: Matplotlib axes to draw on.
        """
