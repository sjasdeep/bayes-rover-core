"""Rover systems with 2D rendering support.

RoverDark: Dubins car with configurable temporally-growing state uncertainty.
RoverBase: Base class with shared 2D rendering utilities (used by RoverLight).
"""

from __future__ import annotations

import math
from itertools import cycle
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
from matplotlib import patches, pyplot as plt
from matplotlib.patches import FancyArrowPatch

from ...core.systems import System
from ...utils.obstacles import Box2D, Circle2D, signed_distance_to_obstacles

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["RoverDark", "RoverBase"]


class RoverBase(System):
    """Base class for Rover systems with shared 2D rendering utilities.
    
    Expects subclasses to define:
        - state_limits: torch.Tensor with shape [2, state_dim]
        - state_labels: tuple of strings
        - obstacles: iterable of obstacle objects
        - goal_state: torch.Tensor
    """

    _render_title: Optional[str] = None  # Override in subclass if desired

    def _get_render_cache(self, ax) -> Dict:
        """Get or create render cache for an axes."""
        if not hasattr(self, "_render_cache"):
            self._render_cache: Dict[int, Dict[str, object]] = {}
        cache = self._render_cache.get(id(ax))
        if cache is None:
            cache = {}
            self._setup_axes(ax)
            cache.update(self._draw_obstacles(ax))
            cache.update(self._draw_reference_markers(ax))
            cache['color_cycle'] = cycle(
                plt.rcParams.get('axes.prop_cycle', plt.cycler(color=['tab:blue'])).by_key().get('color', ['tab:blue'])
            )
            cache['trajectory_count'] = 0
            self._render_cache[id(ax)] = cache
        return cache

    def _setup_axes(self, ax: "Axes") -> None:
        """Configure axes for 2D visualization."""
        title = self._render_title or self.__class__.__name__
        ax.set_title(f'{title} Simulation')
        ax.set_xlabel(self.state_labels[0])
        ax.set_ylabel(self.state_labels[1])
        ax.set_xlim(self.state_limits[:, 0].tolist())
        ax.set_ylim(self.state_limits[:, 1].tolist())
        ax.set_aspect('equal', adjustable='box')
        try:
            ax.set_axisbelow(True)
        except Exception:
            pass
        ax.grid(True, zorder=0)

    def _draw_obstacles(self, ax: "Axes") -> Dict[str, object]:
        """Draw all obstacles on the axes."""
        artists: Dict[str, object] = {}
        for i, obstacle in enumerate(self.obstacles):
            if isinstance(obstacle, Circle2D):
                artist = patches.Circle(
                    obstacle.center.tolist(),
                    radius=float(obstacle.radius),
                    edgecolor='tab:red',
                    facecolor='tab:red',
                    alpha=0.3,
                )
                ax.add_patch(artist)
                artists[f'obstacle_{i}'] = artist
            elif isinstance(obstacle, Box2D):
                cx, cy = obstacle.center.tolist()
                hx, hy = obstacle.half_size.tolist()
                angle = float(torch.rad2deg(obstacle.rotation))
                rectangle = patches.Rectangle(
                    (cx - hx, cy - hy),
                    width=2 * hx,
                    height=2 * hy,
                    angle=angle,
                    edgecolor='tab:orange',
                    facecolor='tab:orange',
                    alpha=0.3,
                )
                ax.add_patch(rectangle)
                artists[f'obstacle_{i}'] = rectangle
        return artists

    def _draw_reference_markers(self, ax: "Axes") -> Dict[str, object]:
        """Draw goal marker and update legend."""
        artists: Dict[str, object] = {}
        goal_artist = ax.plot(
            self.goal_state[0],
            self.goal_state[1],
            marker='*',
            color='tab:green',
            markersize=12,
            label='goal',
            zorder=200,
        )[0]
        artists['goal'] = goal_artist
        if not ax.get_legend():
            ax.legend(loc='upper right')
        return artists

    def _create_base_artists(self, ax, cache, *, include_point: bool = True) -> Dict[str, object]:
        """Create common trajectory artists (line, heading, estimated line/heading)."""
        color = next(cache['color_cycle'])
        label = 'trajectory' if cache['trajectory_count'] == 0 else None
        line, = ax.plot([], [], '-', lw=2, color=color, label=label)
        
        est_label = 'estimated' if cache['trajectory_count'] == 0 else None
        est_line, = ax.plot([], [], '--', lw=1.5, color='gray', alpha=0.9, label=est_label)
        
        heading = FancyArrowPatch((0, 0), (0, 0), color=color, mutation_scale=10)
        ax.add_patch(heading)
        
        est_heading = FancyArrowPatch((0, 0), (0, 0), color='gray', mutation_scale=10, alpha=0.9)
        ax.add_patch(est_heading)
        
        artists = {
            'line': line,
            'est_line': est_line,
            'heading': heading,
            'est_heading': est_heading,
        }
        
        if include_point:
            point, = ax.plot([], [], 'o', color=color, markersize=4)
            artists['point'] = point
        
        cache['trajectory_count'] += 1
        ax.legend(loc='upper right')
        return artists

    def _update_trajectory_lines(self, artists: Dict, history) -> None:
        """Update trajectory line artists from history data."""
        if history is None or 'line' not in artists:
            return
            
        if isinstance(history, dict):
            actual_hist = torch.as_tensor(history.get('actual', torch.empty(0))).detach().cpu()
            if actual_hist.numel() > 0 and actual_hist.ndim >= 2:
                actual_hist = actual_hist.reshape(-1, actual_hist.shape[-1])
                artists['line'].set_data(actual_hist[..., 0], actual_hist[..., 1])
            est_hist = torch.as_tensor(history.get('estimated', torch.empty(0))).detach().cpu()
            if 'est_line' in artists and est_hist.numel() > 0 and est_hist.ndim >= 2:
                est_hist = est_hist.reshape(-1, est_hist.shape[-1])
                artists['est_line'].set_data(est_hist[..., 0], est_hist[..., 1])
        else:
            history_tensor = torch.as_tensor(history).detach().cpu()
            if history_tensor.ndim >= 2:
                history_tensor = history_tensor.reshape(-1, history_tensor.shape[-1])
                artists['line'].set_data(history_tensor[..., 0], history_tensor[..., 1])

    def _update_heading_arrow(self, artists: Dict, state, *, key: str = 'heading', length: float = 0.75) -> None:
        """Update a heading arrow artist."""
        if state is None:
            return
        state_tensor = torch.as_tensor(state).detach().cpu()
        x = float(state_tensor[0])
        y = float(state_tensor[1])
        theta = float(state_tensor[2])
        
        dx = length * math.cos(theta)
        dy = length * math.sin(theta)
        
        arrow = artists.get(key)
        if arrow is not None:
            arrow.set_positions((x, y), (x + dx, y + dy))

    def _update_estimated_heading(self, artists: Dict, history) -> None:
        """Update estimated heading arrow from history dict."""
        if history is None or not isinstance(history, dict) or 'estimated' not in history:
            return
        est_hist = torch.as_tensor(history['estimated']).detach().cpu()
        if est_hist.ndim >= 2 and est_hist.shape[-1] >= 3:
            est_state = est_hist[-1]
            self._update_heading_arrow(artists, est_state, key='est_heading', length=0.6)


class RoverDark(RoverBase):
    """Dubins car with configurable temporally-growing state uncertainty.
    
    State: (x, y, theta) - position and heading.
    Control: omega - angular velocity.
    
    Uncertainty grows linearly from 0 at t=0 to terminal_uncertainty_limits at t=time_horizon,
    scaled by uncertainty_growth_rate. Set uncertainty_growth_rate=0 for nominal (no uncertainty).
    
    Failure function: By default uses signed distance to obstacles. Set failure_grid_value_tag
    to use a precomputed GridValue (BRT) as the failure function instead (for recursive verification).
    """

    state_dim = 3
    state_limits = torch.tensor(
        [
            [0.0, -5.0, -math.pi],      # lower bounds: x, y, theta
            [20.0, 5.0, math.pi],       # upper bounds: x, y, theta
        ],
        dtype=torch.float32,
    )
    state_periodic = [False, False, True]  # x and y are not periodic, theta is periodic
    state_labels = (r'$p_x$ (m)', r'$p_y$ (m)', r'$\theta$ (rad)')

    control_dim = 1
    control_labels = (r'$\omega$ (rad/s)',)

    disturbance_dim = 0
    disturbance_labels = ()

    initial_state = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    # Computational configuration
    _use_gpu = True
    _batch_size = 100000

    # System parameters
    v = 1.0  # constant speed (m/s)
    time_horizon = 5.0  # s
    
    # Uncertainty configuration:
    # - terminal_uncertainty_limits: max uncertainty at t=time_horizon
    # - uncertainty_growth_rate: 0.0 = no uncertainty (nominal), 1.0 = full growth
    terminal_uncertainty_limits: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
        (-0.5, -0.5, -0.1),  # lower bounds
        (0.5, 0.5, 0.1),     # upper bounds
    )
    uncertainty_growth_rate: float = 1.0
    
    @property
    def time_invariant_uncertainty_limits(self) -> bool:
        """True if uncertainty doesn't grow with time (growth_rate == 0)."""
        return self.uncertainty_growth_rate == 0.0
    obstacles = (
        Circle2D(center=(3.0, 2.0), radius=1.0),
        Circle2D(center=(6.0, -1.0), radius=0.3),
        Circle2D(center=(9.0, 0.8), radius=0.5),
        Circle2D(center=(12.0, -1.5), radius=0.7),
        Circle2D(center=(14.0, 1.0), radius=0.4),
        Circle2D(center=(17.0, -0.5), radius=0.6),
        Circle2D(center=(18.5, 1.5), radius=0.4),
    )
    goal_state = torch.tensor([20.0, 0.0, 0.0], dtype=torch.float32)

    # Optional: use a cached GridValue as the failure function (for recursive verification)
    # When set, failure_function queries this GridValue instead of using obstacle SDF.
    failure_grid_value_tag: Optional[str] = None

    def __init__(self) -> None:
        self._render_cache: Dict[int, Dict[str, object]] = {}
        self._failure_grid_value = None  # lazy-loaded GridValue (if failure_grid_value_tag is set)
        self._failure_gv_announced = False

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    def control_limits(self, state, time):
        base_limits = torch.tensor(
            [
                [-1.0],
                [1.0],
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
        terminal_limits = torch.tensor(
            self.terminal_uncertainty_limits,
            dtype=state.dtype,
            device=state.device,
        )
        # Broadcast terminal limits to batch and scale linearly with time
        limits = torch.broadcast_to(terminal_limits, state.shape[:-1] + (2, self.state_dim))
        if self.uncertainty_growth_rate == 0.0:
            # No uncertainty (nominal case)
            return limits[..., 0, :] * 0.0, limits[..., 1, :] * 0.0
        # Scale by (time / time_horizon) * growth_rate, clamped to [0, 1]
        alpha = torch.as_tensor(time, dtype=state.dtype, device=state.device) / float(self.time_horizon)
        alpha = torch.clamp(alpha * self.uncertainty_growth_rate, 0.0, 1.0)
        limits = limits * alpha
        return limits[..., 0, :], limits[..., 1, :]

    # ------------------------------------------------------------------
    # Objective functions
    # ------------------------------------------------------------------
    def failure_function(self, state, time=None):
        # If a GridValue tag is set, use BRT-based failure function
        if self.failure_grid_value_tag is not None:
            return self._failure_function_from_grid_value(state, time)
        # Default: signed distance to obstacles
        return signed_distance_to_obstacles(self.obstacles, state[..., :2].view(-1, 2)).view(*state.shape[:-1])

    def _failure_function_from_grid_value(self, state, time=None):
        """Return value from cached GridValue at t=0 (propagated BRT value)."""
        from ...utils.cache_loaders import load_grid_value_by_tag
        
        if self._failure_grid_value is None:
            tag = self.failure_grid_value_tag
            if not tag:
                raise RuntimeError("failure_grid_value_tag is not set")
            self._failure_grid_value = load_grid_value_by_tag(tag, interpolate=False)
            if not self._failure_gv_announced:
                print(f"Using GridValue for failure_function: tag='{tag}'")
                self._failure_gv_announced = True
        
        gv = self._failure_grid_value
        t0 = float(gv.times[0])  # Query at initial time slice t=0
        st = torch.as_tensor(state)
        return gv.value(st[..., :3], t0, interpolate=True)

    def set_failure_grid_value_tag(self, tag: Optional[str]) -> None:
        """Set or clear the GridValue tag used for failure_function."""
        self.failure_grid_value_tag = tag
        self._failure_grid_value = None
        self._failure_gv_announced = False

    def goal_function(self, state, time=None):
        dx = state[..., 0] - self.goal_state[0]
        dy = state[..., 1] - self.goal_state[1]
        return torch.hypot(dx, dy)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def dynamics(self, state, control, disturbance, time):
        x, y, heading = state.unbind(-1)
        (omega,) = control.unbind(-1)
        x_dot = self.v * torch.cos(heading)
        y_dot = self.v * torch.sin(heading)
        heading_dot = omega
        return torch.stack((x_dot, y_dot, heading_dot), dim=-1)

    # ------------------------------------------------------------------
    # Rendering support
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
            artists = self._create_base_artists(ax, cache, include_point=True)

        self._update_trajectory_lines(artists, history)
        if state is not None:
            state_tensor = torch.as_tensor(state).detach().cpu()
            x, y = float(state_tensor[0]), float(state_tensor[1])
            if 'point' in artists:
                artists['point'].set_data([x], [y])
            self._update_heading_arrow(artists, state)

        self._update_estimated_heading(artists, history)

        return artists


if __name__ == '__main__':

    # visualize system
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2)
    failure_ax, goal_ax = axes

    # create system
    system = RoverDark()
    state_labels = system.state_labels
    state_limits = system.state_limits
    initial_state = system.initial_state
    goal_state = system.goal_state

    # compute grid
    x = torch.linspace(state_limits[0][0], state_limits[1][0], 100)
    y = torch.linspace(state_limits[0][1], state_limits[1][1], 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    pts = torch.stack((X.flatten(), Y.flatten()), dim=-1)

    # plot failure function
    plt.sca(failure_ax)
    sdf = system.failure_function(torch.cat((pts, torch.zeros(len(pts), 1)), dim=-1))
    Z = sdf.reshape(X.shape)
    cp = plt.contourf(X, Y, Z, levels=50, cmap='RdYlBu_r')
    plt.colorbar(cp)
    plt.contour(X, Y, Z, levels=[0.0], colors='k', linewidths=2)

    # plot goal function
    plt.sca(goal_ax)
    gf = system.goal_function(torch.cat((pts, torch.zeros(len(pts), 1)), dim=-1))
    Z = gf.reshape(X.shape)
    cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis_r')
    plt.colorbar(cp)

    # plot goal and initial state in failure plot
    plt.sca(failure_ax)
    plt.plot(goal_state[0], goal_state[1], 'ko', markersize=10)
    plt.plot(initial_state[0], initial_state[1], 'ko', markersize=10)

    # plot goal and initial state in goal plot
    plt.sca(goal_ax)
    plt.plot(goal_state[0], goal_state[1], 'ko', markersize=10)
    plt.plot(initial_state[0], initial_state[1], 'ko', markersize=10)

    # format failure plot
    plt.sca(failure_ax)
    plt.xlim(state_limits[:, 0])
    plt.ylim(state_limits[:, 1])
    plt.xlabel(state_labels[0])
    plt.ylabel(state_labels[1])
    plt.title('RoverDark Failure Function')
    plt.gca().set_aspect('equal', 'box')

    # format goal plot
    plt.sca(goal_ax)
    plt.xlim(state_limits[:, 0])
    plt.ylim(state_limits[:, 1])
    plt.xlabel(state_labels[0])
    plt.ylabel(state_labels[1])
    plt.title('RoverDark Goal Function')
    plt.gca().set_aspect('equal', 'box')

    # show plot
    plt.tight_layout()
    plt.show()
