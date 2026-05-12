"""JAX wrapper for GridInput with JIT-compatible nearest-neighbor lookup.

Enables using GridInput policies inside HJ reachability solvers that run in JAX.
"""

from typing import List

import jax.numpy as jnp
import numpy as np

__all__ = ["JaxGridInput"]


class JaxGridInput:
    """JIT-friendly wrapper for a PyTorch GridInput.

    Preloads grid data into JAX arrays and provides nearest-neighbor lookup
    for deterministic control policies in the HJ solver.
    """

    def __init__(self, torch_grid_input) -> None:
        """Load GridInput data into JAX arrays.

        Args:
            torch_grid_input: A PyTorch GridInput instance with cached grid tensors.
        """
        if torch_grid_input._grid_cache is None:
            raise ValueError("GridInput must have cached grid tensors")

        # Grid axes for state dimensions
        self.grid_axes: List[jnp.ndarray] = [
            self._to_jax(ax) for ax in torch_grid_input._state_grid_points
        ]
        self.state_dim = len(self.grid_axes)
        self.axis_sizes = [int(ax.shape[0]) for ax in self.grid_axes]

        # Time axis (may be None => time-invariant)
        time_pts = getattr(torch_grid_input, "_time_grid_points", None)
        if time_pts is not None:
            self.grid_times = self._to_jax(time_pts)
            self.time_invariant = False
        else:
            self.grid_times = jnp.array([0.0])
            self.time_invariant = True
        self.nt = int(self.grid_times.shape[0])

        # Input values grid: shape [n1, n2, ..., nk, nt, input_dim]
        self.values = self._to_jax(torch_grid_input._grid_cache)
        self.input_dim = int(self.values.shape[-1])

    @staticmethod
    def _to_jax(t):
        """Convert PyTorch tensor to JAX array."""
        if hasattr(t, "detach"):
            return jnp.array(t.detach().cpu().numpy())
        return jnp.array(t)

    def value(self, state: jnp.ndarray, time: float) -> jnp.ndarray:
        """Look up control value at state/time via nearest-neighbor.

        Args:
            state: State vector [D] (unbatched only for HJ solver).
            time: Time scalar.

        Returns:
            Control vector [input_dim].
        """
        # Find nearest grid indices for state
        indices = []
        for d in range(self.state_dim):
            axis = self.grid_axes[d]
            diffs = jnp.abs(axis - state[d])
            idx = jnp.argmin(diffs)
            indices.append(idx)

        # Index into values grid
        # For time-invariant: values shape is [n1, n2, ..., nk, input_dim]
        # For time-varying: values shape is [n1, n2, ..., nk, nt, input_dim]
        v = self.values
        for idx in indices:
            v = v[idx]
        
        # Handle time dimension if present
        if not self.time_invariant:
            t_idx = jnp.argmin(jnp.abs(self.grid_times - time))
            v = v[t_idx]
        
        # v should now have shape [input_dim]
        return v
