"""RoverLight: Dubins car with a light switch that resets uncertainty.

State x = (p_x, p_y, theta, s), where s is time since the light was last ON.
Control u = (omega, light_on), with light_on ∈ {0, 1} (treated as discrete).

Uncertainty bounds grow linearly with s (time since last light was ON) and
reset when the light is turned ON (s := 0).

This system provides a discrete-step `next_state` function in addition to a
continuous `dynamics` for compatibility with existing simulators. The 4th
state uses discrete reset logic, so prefer `next_state` when simulating.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from ...utils.obstacles import Circle2D, signed_distance_to_obstacles
from .rover_dark import RoverBase

__all__ = ["RoverLight"]


class RoverLight(RoverBase):
    """Dubins car with light-controlled uncertainty reset.

    - State: (x, y, theta, s) with s ≥ 0 measuring time since last light ON.
    - Control: (omega, light_on) with light_on in {0, 1}.
    - Disturbance: none by default.
    """

    # ------------------------------------------------------------------
    # State/Control space
    # ------------------------------------------------------------------
    state_dim = 4
    state_limits = torch.tensor(
        [
            [0.0, -5.0, -math.pi, 0.0],   # lower bounds: x, y, theta, s
            [20.0, 5.0, math.pi, 5.0],    # upper bounds: x, y, theta, s_max (default=time_horizon)
        ],
        dtype=torch.float32,
    )
    state_periodic = [False, False, True, False]
    state_labels = (r'$p_x$ (m)', r'$p_y$ (m)', r'$\theta$ (rad)', r'$s$ since light on (s)')

    control_dim = 2
    control_labels = (r'$\omega$ (rad/s)', r'light ON (0/1)')

    disturbance_dim = 0
    disturbance_labels = ()

    initial_state = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    # Uncertainty bounds depend on the state component s
    time_invariant_uncertainty_limits = False

    # Computational configuration
    _use_gpu = True
    _batch_size = 100000

    # System parameters
    v = 1.0  # constant speed (m/s)
    time_horizon = 5.0  # s (also used as default cap for s)
    obstacles = (
        Circle2D(center=(3.0, 2.0), radius=1.0),
        Circle2D(center=(6.0, -1.0), radius=0.3),
        Circle2D(center=(9.0, 0.8), radius=0.5),
        Circle2D(center=(12.0, -1.5), radius=0.7),
        Circle2D(center=(14.0, 1.0), radius=0.4),
        Circle2D(center=(17.0, -0.5), radius=0.6),
        Circle2D(center=(18.5, 1.5), radius=0.4),
    )
    goal_state = torch.tensor([20.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    def __init__(self) -> None:
        # Copy the base limits and ensure s upper bound matches horizon
        self._render_cache: Dict[int, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    def control_limits(self, state, time):
        # omega in [-1, 1], light_on in [0, 1]
        base_limits = torch.tensor(
            [
                [-1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=state.dtype,
            device=state.device,
        )
        limits = torch.broadcast_to(base_limits, state.shape[:-1] + (2, self.control_dim))
        return limits[..., 0, :], limits[..., 1, :]

    def disturbance_limits(self, state, time):
        base_limits = torch.empty(
            (2, self.disturbance_dim),
            dtype=state.dtype,
            device=state.device,
        )
        limits = torch.broadcast_to(base_limits, state.shape[:-1] + (2, self.disturbance_dim))
        return limits[..., 0, :], limits[..., 1, :]

    def uncertainty_limits(self, state, time):
        # Terminal bounds for (x, y, theta); zero for s
        terminal_limits = torch.tensor(
            [
                [-0.5, -0.5, -0.1, 0.0],
                [0.5, 0.5, 0.1, 0.0],
            ],
            dtype=state.dtype,
            device=state.device,
        )
        # Scale linearly with s / time_horizon, clipped to [0, 1]
        s = state[..., 3]
        alpha = torch.clamp(s / float(self.time_horizon), 0.0, 1.0)
        # Broadcast alpha to shape [..., state_dim]
        alpha_expanded = alpha.unsqueeze(-1).expand(*state.shape[:-1], self.state_dim)
        limits = torch.broadcast_to(terminal_limits, state.shape[:-1] + (2, self.state_dim))
        limits = limits * alpha_expanded.unsqueeze(-2)
        return limits[..., 0, :], limits[..., 1, :]

    # ------------------------------------------------------------------
    # Objective functions
    # ------------------------------------------------------------------
    def failure_function(self, state, time=None):
        return signed_distance_to_obstacles(self.obstacles, state[..., :2].view(-1, 2)).view(*state.shape[:-1])

    def goal_function(self, state, time=None):
        dx = state[..., 0] - self.goal_state[0]
        dy = state[..., 1] - self.goal_state[1]
        return torch.hypot(dx, dy)

    # ------------------------------------------------------------------
    # Dynamics (continuous part)
    # ------------------------------------------------------------------
    def dynamics(self, state, control, disturbance, time):
        # Only the Dubins 3D kinematics are expressed continuously; s uses
        # discrete reset logic in `next_state` and remains unchanged here.
        x, y, heading, s = state.unbind(-1)
        omega, light_on = control.unbind(-1)
        x_dot = self.v * torch.cos(heading)
        y_dot = self.v * torch.sin(heading)
        heading_dot = omega
        s_dot = torch.zeros_like(s)
        return torch.stack((x_dot, y_dot, heading_dot, s_dot), dim=-1)

    # ------------------------------------------------------------------
    # Discrete-step transition (preferred for this system)
    # ------------------------------------------------------------------
    def next_state(self, state, control, disturbance, time, dt):
        """One-step update with discrete light-reset.

        - Integrates (x, y, theta) forward by Euler.
        - Updates s := 0 if light_on >= 0.5 else min(s + dt, s_max).
        - Wraps periodic theta and clamps to state limits.
        """
        # Ensure tensors
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state)
        if not isinstance(control, torch.Tensor):
            control = torch.as_tensor(control, dtype=state.dtype, device=state.device)
        if not isinstance(disturbance, torch.Tensor):
            disturbance = torch.as_tensor(disturbance, dtype=state.dtype, device=state.device)

        dtype = state.dtype
        device = state.device

        # Limits on device
        limits = self.state_limits.to(dtype=dtype, device=device)

        # Unpack
        x, y, heading, s = state.unbind(-1)
        omega, light_on = control.unbind(-1)

        # Continuous Dubins update
        x_next = x + dt * (self.v * torch.cos(heading))
        y_next = y + dt * (self.v * torch.sin(heading))
        theta_next = heading + dt * omega

        # Light reset logic for s
        s_max = limits[1, 3]
        is_on = (light_on >= 0.5).to(dtype=dtype)
        s_next = (1.0 - is_on) * torch.clamp(s + dt, min=0.0, max=float(s_max))
        # When light turns ON, s is reset to zero regardless of previous value
        # (captured by multiplication above).

        # Compose and wrap periodic state
        next_state = torch.stack((x_next, y_next, theta_next, s_next), dim=-1)
        # Wrap theta
        low, high = limits[0, 2], limits[1, 2]
        next_state[..., 2] = (next_state[..., 2] - low) % (high - low) + low
        # Clamp to limits
        next_state = torch.clamp(next_state, limits[0], limits[1])
        return next_state

    # ------------------------------------------------------------------
    # Rendering support (extends base mixin with light dots)
    # ------------------------------------------------------------------
    def render(
        self,
        state,
        control,
        disturbance,
        uncertainty,
        time,
        ax,
        *,
        artists: Optional[Dict[str, object]] = None,
        history: Optional[torch.Tensor] = None,
        frame: Optional[int] = None,
    ):
        cache = self._get_render_cache(ax)

        if artists is None:
            # Use base artists and add light-specific ones
            artists = self._create_base_artists(ax, cache, include_point=False)
            # Light indicator: accumulate small dots along the trajectory when light is ON
            light_dots, = ax.plot([], [], 'o', color='gold', markersize=3, linestyle='None')
            artists['light_dots'] = light_dots
            artists['light_dots_x'] = []
            artists['light_dots_y'] = []

        self._update_trajectory_lines(artists, history)

        if state is not None:
            st = torch.as_tensor(state).detach().cpu()
            x, y = float(st[0]), float(st[1])
            self._update_heading_arrow(artists, state)

            # Update light dots when light is ON (accumulates along the path)
            try:
                ctrl = torch.as_tensor(control).detach().cpu()
                light_on = float(ctrl[-1]) if ctrl.numel() > 0 else 0.0
            except Exception:
                light_on = 0.0
            if light_on >= 0.5:
                xs = artists.get('light_dots_x')
                ys = artists.get('light_dots_y')
                if isinstance(xs, list) and isinstance(ys, list):
                    xs.append(x)
                    ys.append(y)
                    ld = artists.get('light_dots')
                    if ld is not None:
                        ld.set_data(xs, ys)

        self._update_estimated_heading(artists, history)

        return artists
