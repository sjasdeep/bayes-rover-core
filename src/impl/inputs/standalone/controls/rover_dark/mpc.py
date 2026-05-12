"""Model Predictive Controller for the :class:`RoverDark` system."""

from __future__ import annotations

import atexit
import multiprocessing as mp
from typing import List, Sequence, Tuple

import casadi as ca
import do_mpc
import numpy as np
import torch

from src.core.inputs import Input
from src.impl.systems.rover_dark import RoverDark
from src.utils.obstacles import Box2D, Circle2D

__all__ = ["RoverDark_MPC"]


# Module-level shared functions for both workers and main class
def _build_dubins_model(speed: float) -> do_mpc.model.Model:
    """Build the do-mpc model for Dubins dynamics (shared by workers and main class)."""
    model = do_mpc.model.Model('continuous')
    
    px = model.set_variable(var_type='_x', var_name='px')
    py = model.set_variable(var_type='_x', var_name='py')
    theta = model.set_variable(var_type='_x', var_name='theta')
    omega = model.set_variable(var_type='_u', var_name='omega')
    
    model.set_rhs('px', speed * ca.cos(theta))
    model.set_rhs('py', speed * ca.sin(theta))
    model.set_rhs('theta', omega)
    model.setup()
    
    return model


def _softplus(value: ca.MX, beta: float = 10.0) -> ca.MX:
    """Smoothed relu used to softly penalize constraint violations."""
    return ca.log1p(ca.exp(beta * value)) / beta


def _circle_signed_distance(px: ca.MX, py: ca.MX, params: Tuple[float, ...]) -> ca.MX:
    """Compute signed distance to circle obstacle."""
    cx, cy, radius = params
    eps = 1e-8  # Prevent gradient singularity at obstacle center
    return ca.sqrt((px - cx) ** 2 + (py - cy) ** 2 + eps) - radius


def _box_signed_distance(px: ca.MX, py: ca.MX, params: Tuple[float, ...]) -> ca.MX:
    """Compute signed distance to box obstacle."""
    cx, cy, hx, hy, angle = params
    dx = px - cx
    dy = py - cy
    
    cos_a = ca.cos(angle)
    sin_a = ca.sin(angle)
    
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    
    dx_abs = ca.fabs(local_x) - hx
    dy_abs = ca.fabs(local_y) - hy
    
    dx_clamped = ca.fmax(dx_abs, 0.0)
    dy_clamped = ca.fmax(dy_abs, 0.0)
    eps = 1e-8  # Prevent gradient singularity when exactly on box edge
    outside_distance = ca.sqrt(dx_clamped ** 2 + dy_clamped ** 2 + eps)
    
    inside_distance = ca.fmin(ca.fmax(dx_abs, dy_abs), 0.0)
    return outside_distance + inside_distance


def _build_collision_penalty(
    px: ca.MX,
    py: ca.MX,
    obstacles: List[Tuple[str, Tuple[float, ...]]],
    obstacle_weight: float,
    obstacle_margin: float
) -> ca.MX:
    """Build collision penalty term for MPC objective (shared by workers and main class)."""
    if not obstacles or obstacle_weight <= 0:
        return ca.MX.zeros(1)
    
    penalties = []
    for kind, params in obstacles:
        if kind == 'circle':
            distance = _circle_signed_distance(px, py, params)
        elif kind == 'box':
            distance = _box_signed_distance(px, py, params)
        else:
            raise ValueError(f'Unknown obstacle type: {kind}')
        
        violation = obstacle_margin - distance
        penalties.append(_softplus(violation))
    
    return obstacle_weight * sum(penalties)


def _build_dubins_mpc(
    model: do_mpc.model.Model,
    dt: float,
    horizon: int,
    control_weight: float,
    control_lower: float,
    control_upper: float,
    goal: Tuple[float, float],
    obstacles: List[Tuple[str, Tuple[float, ...]]],
    obstacle_weight: float,
    obstacle_margin: float
) -> do_mpc.controller.MPC:
    """Build and configure the MPC controller (shared by workers and main class)."""
    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=horizon,
        t_step=dt,
        n_robust=0,
        store_full_solution=False,
        nlpsol_opts={
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
        }
    )
    
    px = model.x['px']
    py = model.x['py']
    omega = model.u['omega']
    
    goal_x, goal_y = goal[0], goal[1]
    base_stage_cost = (px - goal_x) ** 2 + (py - goal_y) ** 2
    collision_penalty = _build_collision_penalty(px, py, obstacles, obstacle_weight, obstacle_margin)
    stage_cost = base_stage_cost + collision_penalty
    terminal_cost = base_stage_cost + collision_penalty
    
    mpc.set_objective(lterm=stage_cost, mterm=terminal_cost)
    mpc.set_rterm(omega=control_weight)
    
    mpc.bounds['lower', '_u', 'omega'] = control_lower
    mpc.bounds['upper', '_u', 'omega'] = control_upper
    
    mpc.setup()
    
    return mpc


# Worker functions for parallel execution
def _worker_init(dt, horizon, control_weight, obstacle_weight, obstacle_margin,
                 state_dim, speed, control_lower, control_upper, goal, obstacles):
    """Initialize worker process with its own MPC instance."""
    global _worker_mpc, _worker_initialized
    
    model = _build_dubins_model(speed)
    _worker_mpc = _build_dubins_mpc(
        model, dt, horizon, control_weight, control_lower, control_upper,
        goal, obstacles, obstacle_weight, obstacle_margin
    )
    _worker_initialized = False


def _worker_compute(state_row):
    """Worker function to compute control for a single state."""
    global _worker_mpc, _worker_initialized
    
    column_state = np.asarray(state_row, dtype=float).reshape(-1, 1)
    _worker_mpc.x0 = column_state
    
    if not _worker_initialized:
        _worker_mpc.set_initial_guess()
        _worker_initialized = True
    
    control = np.asarray(_worker_mpc.make_step(column_state)).reshape(-1)
    return control


class RoverDark_MPC(Input):
    """MPC controller tailored to :class:`src.impl.systems.dubins3d.RoverDark`."""

    type = 'control'
    system_class = RoverDark
    dim = 1  # angular rate control only
    time_invariant = True
    
    # CPU batching configuration (no GPU support)
    _use_gpu = False
    _batch_size = 100000  # Moderate batch size for CPU parallel MPC

    def __init__(
        self,
        dt: float = 0.1,
        horizon: int = 5,
        control_weight: float = 1e-2,
        # Stronger default emphasis on collision avoidance
        obstacle_weight: float = 20.0,
        obstacle_margin: float = 0.5,
        # Robustify avoidance by inflating margin using state uncertainty
        robustify_uncertainty: bool = True,
        robust_margin_factor: float = 1.0,
        num_workers: int = -1,
        parallel_threshold: int = 10,
    ) -> None:
        """Create the MPC controller.
        
        Args:
            dt: Time step for MPC discretization
            horizon: Prediction horizon length
            control_weight: Weight on control effort in objective
            obstacle_weight: Weight on obstacle avoidance penalty
            obstacle_margin: Safety margin around obstacles
            num_workers: Number of parallel workers (default: -1 for auto-detect CPU cores,
                        set to 1 for sequential)
            parallel_threshold: Only use parallel mode for batches >= this size (default: 10)
        """

        self.dt = float(dt)
        self.horizon = int(horizon)
        self.control_weight = float(control_weight)
        self.obstacle_weight = float(obstacle_weight)
        self.obstacle_margin = float(obstacle_margin)
        self.robustify_uncertainty = bool(robustify_uncertainty)
        self.robust_margin_factor = float(robust_margin_factor)
        
        # Set up parallelization
        if num_workers == -1:
            self.num_workers = max(1, mp.cpu_count() - 1)
        else:
            self.num_workers = max(1, int(num_workers))
        self.parallel_threshold = max(1, int(parallel_threshold))

        self._initialised = False
        self._state_dim = RoverDark.state_dim
        self._pool = None
        
        # Register cleanup to avoid shutdown errors
        if self.num_workers > 1:
            atexit.register(self._cleanup_pool)

    def bind(self, system: RoverDark) -> None:
        if not isinstance(system, RoverDark):
            raise TypeError(
                f"RoverDark_MPC requires RoverDark system, "
                f"got {type(system).__name__}"
            )
        
        self._state_dim = system.state_dim
        self._speed = float(system.v)

        initial_state = system.initial_state
        time_tensor = torch.tensor(0.0, dtype=initial_state.dtype, device=initial_state.device)
        lower, upper = system.control_limits(initial_state, time_tensor)
        self._control_lower = lower.detach().cpu().numpy().astype(float).reshape(-1)
        self._control_upper = upper.detach().cpu().numpy().astype(float).reshape(-1)

        self._goal = system.goal_state.detach().cpu().numpy().astype(float)
        self._obstacles = self._extract_obstacle_descriptions(system)

        # Compute an effective safety margin that accounts for state uncertainty
        # over the MPC horizon (conservative: use worst-case terminal uncertainty
        # within horizon and inflate obstacles accordingly).
        effective_margin = self.obstacle_margin
        if self.robustify_uncertainty:
            try:
                t_end = float(min(self.horizon * self.dt, system.time_horizon))
                # Single dummy state to query uncertainty bounds
                dummy_state = system.initial_state.detach().to('cpu')
                lower, upper = system.uncertainty_limits(dummy_state, torch.tensor(t_end, dtype=dummy_state.dtype))
                lower = torch.as_tensor(lower).detach().cpu().numpy()
                upper = torch.as_tensor(upper).detach().cpu().numpy()
                # Half-width per dimension
                half_width = np.maximum(np.abs(lower), np.abs(upper))
                # Inflate by positional uncertainty radius (L2 over x,y)
                rad_xy = float(np.hypot(half_width[0], half_width[1]))
                effective_margin += self.robust_margin_factor * rad_xy
            except Exception:
                # Fallback: keep base margin if uncertainty not available
                pass
        self._effective_margin = float(effective_margin)

        # Always build sequential MPC for single-state queries
        self.model = self._build_model()
        self.mpc = self._build_mpc()
        self._initialised = False
        
        # Also set up worker pool if parallel mode requested
        if self.num_workers > 1:
            if self._pool is not None:
                self._pool.close()
                self._pool.join()
            self._pool = mp.Pool(
                processes=self.num_workers,
                initializer=_worker_init,
                initargs=(self.dt, self.horizon, self.control_weight, self.obstacle_weight,
                         self._effective_margin, self._state_dim, self._speed, 
                         self._control_lower, self._control_upper, self._goal, self._obstacles),
            )

    def _build_model(self) -> do_mpc.model.Model:
        """Construct the do-mpc model describing the Dubins dynamics."""
        return _build_dubins_model(self._speed)

    def _build_mpc(self) -> do_mpc.controller.MPC:
        """Create and configure the do-mpc controller instance."""
        return _build_dubins_mpc(
            self.model,
            self.dt,
            self.horizon,
            self.control_weight,
            self._control_lower,
            self._control_upper,
            self._goal,
            self._obstacles,
            self.obstacle_weight,
            # Use robustified effective margin in the MPC cost
            getattr(self, '_effective_margin', self.obstacle_margin)
        )

    def _extract_obstacle_descriptions(
        self, system: RoverDark
    ) -> List[Tuple[str, Tuple[float, ...]]]:
        """Convert obstacle objects into lightweight tuples for symbolic use."""

        obstacle_specs: List[Tuple[str, Tuple[float, ...]]] = []
        for obstacle in getattr(system, 'obstacles', ()):  # pragma: no branch - simple iteration
            if isinstance(obstacle, Circle2D):
                center = obstacle.center.detach().cpu().numpy().astype(float)
                radius = float(obstacle.radius)
                obstacle_specs.append(('circle', (center[0], center[1], radius)))
            elif isinstance(obstacle, Box2D):
                center = obstacle.center.detach().cpu().numpy().astype(float)
                half_lengths = obstacle.half_size.detach().cpu().numpy().astype(float)
                rotation = float(obstacle.rotation)
                obstacle_specs.append(
                    (
                        'box',
                        (
                            center[0],
                            center[1],
                            half_lengths[0],
                            half_lengths[1],
                            rotation,
                        ),
                    )
                )
            else:  # pragma: no cover - defensive programming
                raise TypeError(
                    'Unsupported obstacle type encountered while building MPC '
                    f'penalty: {type(obstacle)!r}'
                )

        return obstacle_specs

    def _prepare_state_batch(
        self, states: Sequence[Sequence[float]] | torch.Tensor | np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, ...], torch.dtype, torch.device]:
        state_tensor = torch.as_tensor(states)

        if state_tensor.ndim == 0:
            raise ValueError('State input must include the state dimension.')

        if state_tensor.shape[-1] != self._state_dim:
            raise ValueError(
                f'Expected states to have last dimension {self._state_dim}, '
                f'but received shape {tuple(state_tensor.shape)}.'
            )

        batch_shape = tuple(state_tensor.shape[:-1])
        if state_tensor.numel() == 0:
            raise ValueError('Received empty state tensor.')

        dtype = state_tensor.dtype
        device = state_tensor.device
        cpu_tensor = state_tensor.detach().to(dtype=torch.float64, device=torch.device('cpu'))
        flat_states = cpu_tensor.reshape(-1, self._state_dim)
        return flat_states.numpy().astype(float), batch_shape, dtype, device

    def _compute_control(self, state_row: np.ndarray) -> torch.Tensor:
        column_state = np.asarray(state_row, dtype=float).reshape(self._state_dim, 1)

        self.mpc.x0 = column_state

        if not self._initialised:
            self.mpc.set_initial_guess()
            self._initialised = True

        control = np.asarray(self.mpc.make_step(column_state)).reshape(-1)
        return torch.from_numpy(control).to(torch.float32)

    def input(self, state, time):  # type: ignore[override]
        """Compute the optimal control action for the provided state(s)."""

        del time

        if not hasattr(self, 'mpc'):
            raise RuntimeError('Controller must be bound to a system before use.')

        flat_states, batch_shape, dtype, device = self._prepare_state_batch(state)
        batch_size = flat_states.shape[0]

        # Use parallel only if batch is large enough and pool is available
        use_parallel = (self.num_workers > 1 and 
                       batch_size >= self.parallel_threshold and 
                       self._pool is not None)

        if use_parallel:
            # Use multiprocessing pool for large batches
            controls = self._pool.map(_worker_compute, flat_states)
            controls_tensor = torch.from_numpy(np.array(controls)).to(torch.float32)
        else:
            # Sequential execution for small batches or single states
            controls = [self._compute_control(row) for row in flat_states]
            controls_tensor = torch.stack(controls, dim=0)

        result = controls_tensor.reshape(*batch_shape, self.dim)
        return result.to(dtype=dtype, device=device)

    def reset(self) -> None:
        """Reset the internal warm-start state used by the MPC solver."""

        # Reset sequential MPC
        self._initialised = False
        if hasattr(self, 'mpc') and hasattr(self.mpc, 'reset_history'):
            self.mpc.reset_history()
        
        # Recreate parallel pool if it exists
        if self.num_workers > 1 and hasattr(self, '_pool') and self._pool is not None:
            self._pool.close()
            self._pool.join()
            if hasattr(self, '_speed'):  # Only recreate if already bound
                self._pool = mp.Pool(
                    processes=self.num_workers,
                    initializer=_worker_init,
                    initargs=(self.dt, self.horizon, self.control_weight, self.obstacle_weight,
                             getattr(self, '_effective_margin', self.obstacle_margin), self._state_dim, self._speed,
                             self._control_lower, self._control_upper, self._goal, self._obstacles),
                )

    def _cleanup_pool(self):
        """Clean up the worker pool safely."""
        if hasattr(self, '_pool') and self._pool is not None:
            try:
                self._pool.terminate()  # Faster than close() during shutdown
                self._pool.join()
                self._pool = None
            except Exception:
                pass  # Ignore all errors during cleanup

    def __del__(self):
        """Clean up the worker pool on deletion."""
        self._cleanup_pool()
