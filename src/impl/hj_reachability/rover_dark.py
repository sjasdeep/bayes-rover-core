from typing import Tuple, Any

import jax.numpy as jnp
import torch

from ...core.hj_reachability import ChannelConfig, ChannelMode, HJReachabilityDynamics
from ...core.inputs import Input
from ...core.sets import Set
from ..inputs.jax_grid_input import JaxGridInput
from ..sets.jax_grid_set import JaxGridSet
from ..systems.rover_dark import RoverDark as RoverDarkSystem
from .base import HJSolverDynamics

__all__ = ["RoverDark", "RoverDarkNominal", "RuntimeRoverDark"]


class RuntimeRoverDark(HJReachabilityDynamics):
    """RoverDark runtime dynamics using PyTorch GridSet for uncertainty optimization."""

    def optimal_uncertainty_from_grad(self, state, time, grad):
        if self.control.given_kind != "set" or self.control.given_set is None:
            raise RuntimeError("Control set must be bound before extracting optimal uncertainty.")
        state_t = torch.as_tensor(state).to(torch.float32)
        grad_t = torch.as_tensor(grad, dtype=state_t.dtype, device=state_t.device)
        flat_state = state_t.reshape(-1, state_t.shape[-1])
        direction = -grad_t.reshape(-1, grad_t.shape[-1])[:, 2:3]  # dV/dtheta
        control_set = self.control.given_set.to(state_t.device)
        _, xhat = control_set.argmax_support_with_state_est(direction, flat_state, float(time))
        return (xhat.reshape_as(flat_state) - flat_state).reshape_as(state_t)


class _RoverDarkBase(HJSolverDynamics):
    """Shared base for RoverDark HJ dynamics variants.
    
    Provides common structure for unicycle dynamics with heading control.
    Subclasses implement hamiltonian() and _get_control_bounds() differently.
    """

    def __init__(self) -> None:
        self.system = RoverDarkSystem()
        self._v = self.system.v

    def __call__(self, state: jnp.ndarray, control: jnp.ndarray, disturbance: jnp.ndarray, time: float) -> jnp.ndarray:
        """Dynamics evaluation (not used by solver, but required by interface)."""
        raise NotImplementedError("Use hamiltonian() method instead")

    def optimal_control_and_disturbance(
        self, 
        state: jnp.ndarray, 
        time: float, 
        grad_value: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Not used since we override hamiltonian() directly."""
        raise NotImplementedError("Use hamiltonian() method instead")

    def _get_control_bounds(self, state: jnp.ndarray, time: float) -> Tuple[float, float]:
        """Return (omega_min, omega_max) for dissipation coefficient computation."""
        raise NotImplementedError("Subclass must implement _get_control_bounds")

    def partial_max_magnitudes(
        self, 
        state: jnp.ndarray, 
        time: float, 
        value: jnp.ndarray,
        grad_value_box: Any
    ) -> jnp.ndarray:
        """Computes max magnitudes of Hamiltonian partials over grad_value_box.
        
        For unicycle dynamics H = dVdx*v*cos(θ) + dVdy*v*sin(θ) + dVdtheta*ω:
            ∂H/∂(dVdx) = v*cos(θ)
            ∂H/∂(dVdy) = v*sin(θ)  
            ∂H/∂(dVdtheta) = ω (bounded by control limits)
        """
        theta = state[2]
        
        partial_x_mag = jnp.abs(self._v * jnp.cos(theta))
        partial_y_mag = jnp.abs(self._v * jnp.sin(theta))
        
        omega_min, omega_max = self._get_control_bounds(state, time)
        partial_theta_mag = jnp.max(jnp.abs(jnp.array([omega_min, omega_max])))

        return jnp.array([partial_x_mag, partial_y_mag, partial_theta_mag])


class RoverDark(_RoverDarkBase):
    """HJ reachability dynamics for RoverDark with observation-based control.
    
    This class implements the closed-loop dynamics where:
    - The agent computes control ω from estimated state x̂ (stored in GridSet)
    - State estimation error e = x̂ - x acts as the adversarial disturbance
    """

    @classmethod
    def runtime_class(cls) -> type:
        return RuntimeRoverDark
    
    def __init__(self) -> None:
        super().__init__()
        self.control = ChannelConfig(mode=ChannelMode.GIVEN)
        self.disturbance = ChannelConfig(mode=ChannelMode.ZERO)
        self.uncertainty = ChannelConfig(mode=ChannelMode.OPTIMIZE)
        self.control.given_kind = 'set'
    
    def _get_control_bounds(self, state: jnp.ndarray, time: float) -> Tuple[float, float]:
        """Get control bounds from the GridSet at this state/time."""
        lower_ctrl, upper_ctrl = self.jax_grid_set.as_box(state, time)
        return lower_ctrl[0], upper_ctrl[0]

    def hamiltonian(self, state: jnp.ndarray, time: float, value: jnp.ndarray, grad_value: jnp.ndarray) -> jnp.ndarray:
        """Compute the Hamiltonian for worst-case uncertainty problem.
        
        For worst-case control (min over control set), we want H = min_u ∇V · f(x, u).
        This is equivalent to argmax in direction [-dVdtheta] to minimize the dot product.
        """
        dVdx, dVdy, dVdtheta = grad_value[0], grad_value[1], grad_value[2]
        theta = state[2]
        
        # For worst-case control: minimize H = ∇V · f = dVdtheta * omega
        optimal_omega = self.jax_grid_set.argmax_support(jnp.array([-dVdtheta]), state, time)[0]
        
        # H = ∇V · f
        return dVdx * self._v * jnp.cos(theta) + dVdy * self._v * jnp.sin(theta) + dVdtheta * optimal_omega

    # -------------------- HJReachabilityDynamics binding interface --------------------
    def bind_control_set(self, given_set: Set) -> None:
        """Bind a PyTorch GridSet for control and create a JAX wrapper."""
        self.control.given_set = given_set
        self.jax_grid_set = JaxGridSet(given_set)

    # -------------------- Optimal channel extraction --------------------
    def optimal_uncertainty_from_grad(self, state, time, grad_value):
        """Return the state-estimation error that induces the worst-case control."""
        if not hasattr(self, "jax_grid_set"):
            raise RuntimeError("Control set must be bound before extracting optimal uncertainty.")

        import numpy as np
        import torch

        def _to_numpy(arr):
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
            return np.asarray(arr)

        state_np = _to_numpy(state)
        grad_np = _to_numpy(grad_value)

        original_shape = state_np.shape
        state_flat = state_np.reshape(-1, state_np.shape[-1])
        grad_flat = grad_np.reshape(-1, grad_np.shape[-1])

        if grad_flat.shape[-1] < 3:
            raise ValueError("Gradient must include heading derivative to recover uncertainty.")

        direction = -grad_flat[:, 2:3]  # adversary pushes against dV/dtheta

        state_jnp = jnp.asarray(state_flat)
        dir_jnp = jnp.asarray(direction)

        best_u, best_xhat, has_state = self.jax_grid_set.argmax_support_with_state_est(
            dir_jnp,
            state_jnp,
            float(time),
        )

        best_xhat = np.asarray(best_xhat)
        has_state = np.asarray(has_state)

        uncertainty = np.zeros_like(state_flat)
        if best_xhat.ndim == 1:
            has = bool(has_state) if np.ndim(has_state) == 0 else bool(has_state[0])
            if has:
                uncertainty = (best_xhat - state_flat).reshape(uncertainty.shape)
        else:
            mask = has_state.astype(bool)
            if mask.ndim == 0:
                mask = np.array([bool(mask)])
            if mask.any():
                diffs = best_xhat - state_flat
                uncertainty[mask] = diffs[mask]

        return uncertainty.reshape(original_shape)


class RoverDarkNominal(_RoverDarkBase):
    """Nominal HJ dynamics for RoverDark with deterministic control (no uncertainty).
    
    This class computes the BRT for a given deterministic control policy (GridInput).
    Unlike the robust RoverDark dynamics, there is no adversarial uncertainty.
    """

    def __init__(self, system=None) -> None:
        # Accept optional system arg for compatibility with runtime_class() interface;
        # _RoverDarkBase creates its own system instance internally.
        super().__init__()
        self.control = ChannelConfig(mode=ChannelMode.GIVEN)
        self.disturbance = ChannelConfig(mode=ChannelMode.ZERO)
        self.uncertainty = ChannelConfig(mode=ChannelMode.ZERO)
        self.control.given_kind = 'input'
        self._jax_grid_input = None

    def _get_control_bounds(self, state: jnp.ndarray, time: float) -> Tuple[float, float]:
        """RoverDark control limits are fixed at [-1.0, 1.0] rad/s."""
        return -1.0, 1.0

    def bind_control_input(self, given_input: Input) -> None:
        """Bind a GridInput for deterministic control and create JAX wrapper."""
        self.control.given_input = given_input
        self._jax_grid_input = JaxGridInput(given_input)

    def hamiltonian(self, state: jnp.ndarray, time: float, value: jnp.ndarray, grad_value: jnp.ndarray) -> jnp.ndarray:
        """Compute Hamiltonian for nominal (no uncertainty) dynamics.
        
        H = ∇V · f(x, u(x)) where u(x) is the deterministic control from GridInput.
        """
        dVdx, dVdy, dVdtheta = grad_value[0], grad_value[1], grad_value[2]
        theta = state[2]
        
        # Get deterministic control from JAX wrapper
        omega = self._jax_grid_input.value(state, time)[0]
        
        # H = ∇V · f
        return dVdx * self._v * jnp.cos(theta) + dVdy * self._v * jnp.sin(theta) + dVdtheta * omega
