"""Control for RoverLight: reuse RoverDark yaw controller + schedule light via GridValue.

This input composes:
  - Yaw control (omega): uses the same RoverDark controller (MPC) by binding an
    internal instance to a shadow RoverDark system mirroring key params from
    RoverLight. State is projected to (x,y,theta) for the inner control.
  - Light control (binary): computed from a cached GridValue (RoverDark BRT).

Scheduling rule:
  The rover only knows its true state at the start and immediately after turning on
  the light (which resets estimation error to 0). At each such reset (s == 0):
    1. Compute t* = max { t in [0, H] | V(x3, t) <= 0 } for the known state x3.
    2. Store d = H - t* (delay until light must turn ON).
    3. As uncertainty clock s grows, turn light ON when s >= d.
  If no t satisfies V <= 0, light stays OFF (state is never in the BRT).

  After the light turns on, s resets to 0, the state is known again, and the
  schedule is recomputed. This policy is stateful: the scheduled delay d is
  locked in at each reset and held until the next light-on event.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.core.inputs import Input
from src.impl.systems.rover_light import RoverLight
from src.impl.systems.rover_dark import RoverDark
from src.utils.cache_loaders import load_grid_value_by_tag

__all__ = ["RoverLight_BRTScheduledLight"]


class RoverLight_BRTScheduledLight(Input):
    """RoverLight control: RoverDark yaw + light from RoverDark GridValue.

    Args:
        grid_value_tag: Cache tag for the RoverDark GridValue (worst-case uncertainty BRT).
        interpolate: Whether to use nearest-neighbour interpolation when querying the grid.
        # MPC yaw params (mirrors RoverDark_MPC defaults)
        mpc_dt: Discretization timestep used by the MPC controller.
        mpc_horizon: Prediction horizon (steps).
        mpc_control_weight: Weight on control effort.
        mpc_obstacle_weight: Collision penalty weight.
        mpc_obstacle_margin: Base safety margin around obstacles.
        mpc_robustify_uncertainty: Inflate margin using terminal uncertainty over horizon.
        mpc_robust_margin_factor: Scale factor for robust inflation.
    """

    type = 'control'
    system_class = RoverLight
    dim = 2  # (omega, light_on)
    time_invariant = False

    def __init__(
        self,
        *,
        grid_value_tag: Optional[str] = None,
        interpolate: bool = True,
        mpc_dt: float = 0.1,
        mpc_horizon: int = 5,
        mpc_control_weight: float = 1e-2,
        mpc_obstacle_weight: float = 20.0,
        mpc_obstacle_margin: float = 0.5,
        mpc_robustify_uncertainty: bool = True,
        mpc_robust_margin_factor: float = 1.0,
    ) -> None:
        self._grid_tag = (str(grid_value_tag) if grid_value_tag is not None else None)
        self._interpolate = bool(interpolate)

        # Store MPC params; actual controller built on bind()
        self._mpc_dt = float(mpc_dt)
        self._mpc_horizon = int(mpc_horizon)
        self._mpc_control_weight = float(mpc_control_weight)
        self._mpc_obstacle_weight = float(mpc_obstacle_weight)
        self._mpc_obstacle_margin = float(mpc_obstacle_margin)
        self._mpc_robustify_uncertainty = bool(mpc_robustify_uncertainty)
        self._mpc_robust_margin_factor = float(mpc_robust_margin_factor)

        # Lazy members
        self._system: Optional[RoverLight] = None
        self._inner_yaw = None  # RoverDark_MPC instance
        self._grid_value = None
        self._times_np: Optional[np.ndarray] = None
        self._times_t: Optional[torch.Tensor] = None
        self.dim = 2
        # Per-batch scheduled delays d = H - t* (initialized to +inf => no trigger)
        self._scheduled_d = None  # shape [N] on first call
        self._scheduled_batch = None

    # ------------------------------------------------------------------
    # Input interface
    # ------------------------------------------------------------------
    def bind(self, system: RoverLight) -> None:
        if not isinstance(system, RoverLight):
            raise TypeError(
                f"{type(self).__name__} requires RoverLight system, got {type(system).__name__}"
            )
        self._system = system

        # 1) Build an inner RoverDark controller and bind to a shadow RoverDark
        from src.impl.inputs.standalone.controls.rover_dark.mpc import RoverDark_MPC

        shadow = RoverDark()
        # Mirror key parameters from 4DLight to maintain behaviour
        shadow.v = float(system.v)
        shadow.obstacles = tuple(system.obstacles)
        shadow.goal_state = torch.as_tensor([float(system.goal_state[0]), float(system.goal_state[1]), 0.0], dtype=torch.float32)
        shadow.state_limits = torch.tensor(
            [
                [float(system.state_limits[0, 0]), float(system.state_limits[0, 1]), -np.pi],
                [float(system.state_limits[1, 0]), float(system.state_limits[1, 1]),  np.pi],
            ],
            dtype=torch.float32,
        )

        inner = RoverDark_MPC(
            dt=self._mpc_dt,
            horizon=self._mpc_horizon,
            control_weight=self._mpc_control_weight,
            obstacle_weight=self._mpc_obstacle_weight,
            obstacle_margin=self._mpc_obstacle_margin,
            robustify_uncertainty=self._mpc_robustify_uncertainty,
            robust_margin_factor=self._mpc_robust_margin_factor,
        )
        inner.set_type('control')
        inner.bind(shadow)
        self._inner_yaw = inner

        # 2) Resolve and load GridValue for scheduling
        grid_tag = self._grid_tag
        if grid_tag is None:
            # Try simulation config fallback (only if not provided via constructor)
            try:
                from src.utils.config import load_simulation_config
                # Load system-level config with preset support
                cfg = load_simulation_config(type(system).__name__)
                grid_tag = cfg.get('control_grid_value_tag')
            except Exception:
                grid_tag = None
        if not grid_tag:
            raise ValueError(
                "GridValue tag is required. Provide grid_value_tag in constructor or set "
                "config/simulations.yaml under this control with key 'grid_value_tag'."
            )
        gv = load_grid_value_by_tag(grid_tag, interpolate=self._interpolate)
        self._grid_value = gv
        self._times_np = np.asarray(gv.times, dtype=float)
        if self._times_np.size == 0:
            raise ValueError("GridValue has empty time axis.")
        # Cache as torch for vectorised queries
        self._times_t = torch.as_tensor(self._times_np, dtype=torch.float32)

        # Sanity: ensure horizon compatibility (within small tolerance)
        H = float(system.time_horizon)
        if not (abs(float(self._times_np[0]) - 0.0) < 1e-6 and abs(float(self._times_np[-1]) - H) < 1e-3):
            # Not fatal; warn via print for now (caller-facing scripts print stdout)
            print(
                f"[Warn] GridValue time domain [{self._times_np[0]:.3f}, {self._times_np[-1]:.3f}] "
                f"differs from system horizon [0, {H:.3f}] — proceeding"
            )

        self.dim = 2

    def to(self, device: torch.device | str):  # pragma: no cover
        # GridValue queries are CPU-based; inner MPC is CPU as well. No-op.
        return self

    def input(self, state: torch.Tensor, time: float) -> torch.Tensor:
        if self._system is None or self._inner_yaw is None or self._grid_value is None:
            raise RuntimeError("bind() must be called before using the control input.")

        state = torch.as_tensor(state)
        orig_dtype = state.dtype
        orig_device = state.device

        # Project to RoverDark for yaw control (always available)
        x3 = state[..., :3]
        omega = torch.as_tensor(self._inner_yaw.input(x3, time), dtype=orig_dtype, device=orig_device)
        omega = omega.reshape(*state.shape[:-1], 1)

        # Light scheduling via GridValue (compute t* ONLY at resets when s == 0)
        s_cpu = state[..., 3].detach().cpu().to(torch.float32).reshape(-1)
        x3_cpu = x3.detach().cpu().to(torch.float32).reshape(-1, 3)

        times = self._times_t  # [T]
        N = int(x3_cpu.shape[0])
        T = int(times.numel())

        # Initialize per-batch schedule storage if shape changed
        if self._scheduled_d is None or self._scheduled_batch != N:
            self._scheduled_d = torch.full((N,), float('inf'), dtype=torch.float32)
            self._scheduled_batch = N

        # Find indices that are at reset (s == 0): recompute schedule only for these
        at_reset = (s_cpu == 0.0)
        if at_reset.any():
            idx = torch.nonzero(at_reset, as_tuple=False).reshape(-1)
            x_sub = x3_cpu.index_select(0, idx)
            n_sub = int(x_sub.shape[0])
            # Repeat along times and query V(x_sub, t)
            x_rep = x_sub.unsqueeze(1).expand(n_sub, T, 3).reshape(n_sub * T, 3)
            t_rep = times.unsqueeze(0).expand(n_sub, T).reshape(n_sub * T)
            v = self._grid_value.value(x_rep, t_rep).reshape(n_sub, T)
            mask = v <= 0.0
            # Default no BRT membership ⇒ d = +inf
            t_star_idx = torch.full((n_sub,), -1, dtype=torch.int64)
            any_true = mask.any(dim=1)
            if any_true.any():
                # Find last True along time by flipping then argmax
                flipped = torch.flip(mask.int(), dims=[1])
                idx_last = flipped.argmax(dim=1)
                forward_idx = (T - 1) - idx_last
                t_star_idx = torch.where(any_true, forward_idx, t_star_idx)
            # Compute d = H - t*
            H = float(self._system.time_horizon)
            d_sub = torch.full((n_sub,), float('inf'), dtype=torch.float32)
            valid = t_star_idx >= 0
            if valid.any():
                t_star = times[t_star_idx[valid]]
                d_sub[valid] = H - t_star
            # Update schedules at the reset indices
            self._scheduled_d.index_copy_(0, idx, d_sub)

        # Decide light based on current s and scheduled delay
        d = self._scheduled_d  # shape [N], float32 CPU
        light = (s_cpu >= d).to(torch.float32).reshape(*state.shape[:-1], 1)
        light = light.to(dtype=orig_dtype, device=orig_device)

        return torch.cat((omega, light), dim=-1)
