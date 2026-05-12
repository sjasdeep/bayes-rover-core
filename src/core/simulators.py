"""Simulation utilities for closed-loop system rollouts.

This module provides simulators for integrating dynamical systems under
control, disturbance, and state uncertainty. The key semantic is that the
controller observes a noisy state estimate (true state + uncertainty).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from tqdm import tqdm

from .inputs import Input
from .systems import System

__all__ = ["SimulationResult", "simulate_euler", "simulate_discrete"]


@dataclass
class SimulationResult:
    """Container for simulation rollout data.

    All tensor attributes have batch dimensions ``[..., T, dim]`` where T is
    the number of time steps (T+1 for states, T for inputs).

    Attributes:
        system: The System instance that was simulated.
        system_name: Class name of the system.
        states: State trajectory, shape ``[..., num_steps+1, state_dim]``.
        controls: Control inputs, shape ``[..., num_steps, control_dim]``.
        disturbances: Disturbance inputs, shape ``[..., num_steps, disturbance_dim]``.
        uncertainties: State estimation errors, shape ``[..., num_steps, state_dim]``.
        estimated_states: Observed states (true + uncertainty), shape ``[..., num_steps, state_dim]``.
        times: Time values, shape ``[num_steps+1]``.
    """

    system: Optional[System] = None
    system_name: str = ""
    states: Optional[torch.Tensor] = None
    controls: Optional[torch.Tensor] = None
    disturbances: Optional[torch.Tensor] = None
    uncertainties: Optional[torch.Tensor] = None
    estimated_states: Optional[torch.Tensor] = None
    times: Optional[torch.Tensor] = None

    def save(self, filepath: str) -> None:
        """Save the simulation result to a pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> SimulationResult:
        """Load a simulation result from a pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_device(
    value: Union[torch.Tensor, float],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Ensure value is a tensor on the correct device."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _wrap_periodic(
    state: torch.Tensor,
    periodic: list[bool],
    limits: torch.Tensor,
) -> torch.Tensor:
    """Wrap periodic state dimensions to their canonical range (in-place)."""
    for dim, is_periodic in enumerate(periodic):
        if is_periodic:
            lo, hi = limits[0, dim], limits[1, dim]
            state[..., dim] = (state[..., dim] - lo) % (hi - lo) + lo
    return state


def _simulate(
    system: System,
    control: Input,
    disturbance: Input,
    uncertainty: Input,
    dt: float,
    num_steps: int,
    initial_state: torch.Tensor,
    step_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float, float], torch.Tensor],
    show_progress: bool,
    leave_progress: bool,
    enforce_constraints: bool,
    device: Optional[torch.device],
) -> SimulationResult:
    """Core simulation loop (internal).

    Args:
        step_fn: Function (state, control, disturbance, time, dt) -> next_state.
    """
    # Reset stateful inputs (e.g., recurrent policies)
    control.reset()
    disturbance.reset()
    uncertainty.reset()

    # Setup dtype/device
    if initial_state.ndim == 1:
        initial_state = initial_state.unsqueeze(0)

    dtype = initial_state.dtype
    dev = device if device is not None else initial_state.device
    if initial_state.device != dev:
        initial_state = initial_state.to(device=dev)

    limits = system.state_limits.to(device=dev, dtype=dtype)
    periodic = system.state_periodic

    # Get dimensions
    batch = initial_state.shape[:-1]
    batch_size = batch[0] if batch else 1
    state_dim = initial_state.shape[-1]
    ctrl_dim = getattr(control, "dim", system.control_dim)
    dist_dim = getattr(disturbance, "dim", system.disturbance_dim)
    unc_dim = getattr(uncertainty, "dim", state_dim)

    # Allocate storage
    states = torch.zeros((*batch, num_steps + 1, state_dim), dtype=dtype, device=dev)
    controls = torch.zeros((*batch, num_steps, ctrl_dim), dtype=dtype, device=dev)
    disturbances = torch.zeros((*batch, num_steps, dist_dim), dtype=dtype, device=dev)
    uncertainties = torch.zeros((*batch, num_steps, unc_dim), dtype=dtype, device=dev)
    estimated_states = torch.zeros((*batch, num_steps, state_dim), dtype=dtype, device=dev)
    times = torch.zeros(num_steps + 1, dtype=dtype, device=dev)

    # Initialize state
    state = _wrap_periodic(initial_state.clone(), periodic, limits)
    if enforce_constraints:
        state = torch.clamp(state, limits[0], limits[1])
    states[..., 0, :] = state

    # Simulation loop
    for step in tqdm(range(num_steps), disable=not show_progress, leave=leave_progress):
        t = step * dt

        # Compute uncertainty and state estimate
        e = _to_device(uncertainty.input(state, t), dtype, dev)
        if enforce_constraints:
            e_lo, e_hi = system.uncertainty_limits(state, t)
            e = torch.clamp(e, _to_device(e_lo, dtype, dev), _to_device(e_hi, dtype, dev))
        x_est = _wrap_periodic(state + e, periodic, limits)

        # Compute control (from estimate)
        u = _to_device(control.input(x_est, t), dtype, dev)

        # Compute disturbance (from true state)
        d = _to_device(disturbance.input(state, t), dtype, dev)

        if enforce_constraints:
            u_lo, u_hi = system.control_limits(state, t)
            d_lo, d_hi = system.disturbance_limits(state, t)
            u = torch.clamp(u, _to_device(u_lo, dtype, dev), _to_device(u_hi, dtype, dev))
            d = torch.clamp(d, _to_device(d_lo, dtype, dev), _to_device(d_hi, dtype, dev))

        # Store
        controls[..., step, :] = u
        disturbances[..., step, :] = d
        uncertainties[..., step, :] = e
        estimated_states[..., step, :] = x_est

        # Advance state
        state = step_fn(state, u, d, t, dt)
        state = _wrap_periodic(state, periodic, limits)
        if enforce_constraints:
            state = torch.clamp(state, limits[0], limits[1])

        states[..., step + 1, :] = state
        times[step + 1] = t + dt

    return SimulationResult(
        system=system,
        system_name=type(system).__name__,
        states=states,
        controls=controls,
        disturbances=disturbances,
        uncertainties=uncertainties,
        estimated_states=estimated_states,
        times=times,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_euler(
    system: System,
    control: Input,
    disturbance: Input,
    uncertainty: Input,
    dt: float,
    num_steps: int,
    initial_state: torch.Tensor,
    show_progress: bool = False,
    leave_progress: bool = False,
    enforce_system_constraints: bool = True,
    device: Optional[Union[torch.device, str]] = None,
) -> SimulationResult:
    """Simulate using forward Euler integration.

    Integrates ``dx/dt = f(x, u, d, t)`` via ``x_{k+1} = x_k + dt * f(x_k, u_k, d_k, t_k)``.

    Args:
        system: The dynamical system to simulate.
        control: Control input policy (receives state estimate).
        disturbance: Disturbance input policy (receives true state).
        uncertainty: Uncertainty input policy (state estimation error).
        dt: Time step size.
        num_steps: Number of simulation steps.
        initial_state: Initial state, shape ``[state_dim]`` or ``[batch, state_dim]``.
        show_progress: Display a progress bar.
        leave_progress: Keep progress bar visible after completion.
        enforce_system_constraints: Clamp inputs to system limits.
        device: Target device (None uses initial_state's device).

    Returns:
        SimulationResult with the full trajectory.
    """
    if isinstance(device, str):
        device = torch.device(device)

    dtype = initial_state.dtype
    dev = device if device is not None else initial_state.device

    def euler_step(state, u, d, t, dt):
        xdot = _to_device(system.dynamics(state, u, d, t), dtype, dev)
        return state + dt * xdot

    return _simulate(
        system, control, disturbance, uncertainty, dt, num_steps, initial_state,
        euler_step, show_progress, leave_progress, enforce_system_constraints, device,
    )


def simulate_discrete(
    system: System,
    control: Input,
    disturbance: Input,
    uncertainty: Input,
    dt: float,
    num_steps: int,
    initial_state: torch.Tensor,
    show_progress: bool = False,
    leave_progress: bool = False,
    enforce_system_constraints: bool = True,
    device: Optional[Union[torch.device, str]] = None,
) -> SimulationResult:
    """Simulate using discrete state transitions.

    Uses ``system.next_state(x, u, d, t, dt)`` if available, else falls back to Euler.

    Args:
        system: The dynamical system to simulate.
        control: Control input policy (receives state estimate).
        disturbance: Disturbance input policy (receives true state).
        uncertainty: Uncertainty input policy (state estimation error).
        dt: Time step size.
        num_steps: Number of simulation steps.
        initial_state: Initial state, shape ``[state_dim]`` or ``[batch, state_dim]``.
        show_progress: Display a progress bar.
        leave_progress: Keep progress bar visible after completion.
        enforce_system_constraints: Clamp inputs to system limits.
        device: Target device (None uses initial_state's device).

    Returns:
        SimulationResult with the full trajectory.
    """
    if isinstance(device, str):
        device = torch.device(device)

    dtype = initial_state.dtype
    dev = device if device is not None else initial_state.device

    next_state_fn = getattr(system, "next_state", None)
    if callable(next_state_fn):
        def discrete_step(state, u, d, t, dt):
            return _to_device(next_state_fn(state, u, d, t, dt), dtype, dev)
    else:
        # Fall back to Euler
        def discrete_step(state, u, d, t, dt):
            xdot = _to_device(system.dynamics(state, u, d, t), dtype, dev)
            return state + dt * xdot

    return _simulate(
        system, control, disturbance, uncertainty, dt, num_steps, initial_state,
        discrete_step, show_progress, leave_progress, enforce_system_constraints, device,
    )
