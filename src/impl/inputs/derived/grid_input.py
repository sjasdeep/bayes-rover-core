"""
Grid-based input wrapper for querying precomputed values.

This module provides a lightweight wrapper for evaluating
an input on a precomputed grid of states (and optionally time).
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch

from src.core.inputs import Input
from src.core.systems import System
from src.utils.grids import (
    exact_axis_indices as _exact_axis_indices_util,
    exact_state_indices as _exact_state_indices_util,
    nearest_state_indices as _nearest_state_indices_util,
    nearest_time_index as _nearest_time_index_util,
)

__all__ = ["GridInput"]


class GridInput(Input):
    """
    A grid-based input wrapper for querying precomputed values.

    This class assumes the grid tensors (state/time points and the cached input values)
    are provided at construction time.
    """
    system_class = System
    type = 'any'
    time_invariant = False
    
    def __init__(
        self,
        wrapped_input: Optional[Input] = None,
        *,
        grid_cache: Optional[torch.Tensor] = None,
        state_grid_points: Optional[List[torch.Tensor]] = None,
        time_grid_points: Optional[torch.Tensor] = None,
        interpolate: bool = False,
    ):
        """Initialize the wrapper for querying a precomputed grid."""
        self.wrapped_input = wrapped_input
        self.interpolate = interpolate

        # Initialize grid storage
        self._grid_cache: Optional[torch.Tensor] = grid_cache
        self._state_grid_points: Optional[List[torch.Tensor]] = state_grid_points
        self._time_grid_points: Optional[torch.Tensor] = time_grid_points
        self._system: Optional[System] = None

        # Device management (cache stored on CPU by default, can be moved to GPU)
        self._cache_device = torch.device('cpu')

        # Delegate attributes from wrapped_input if present
        if wrapped_input is not None:
            self.type = getattr(wrapped_input, 'type', 'any')
            self.system_class = getattr(wrapped_input, 'system_class', System)
            self.time_invariant = getattr(wrapped_input, 'time_invariant', False)
            self.dim = getattr(wrapped_input, 'dim', None)
        else:
            # Infer from provided tensors when possible
            if grid_cache is not None:
                self.dim = int(grid_cache.shape[-1])
            self.time_invariant = time_grid_points is None

    def set_type(self, type: Literal['any', 'control', 'disturbance', 'uncertainty']) -> None:
        """Set the input type (and propagate to wrapped input if present)."""
        super().set_type(type)
        if self.wrapped_input is not None:
            self.wrapped_input.set_type(type)
    
    def bind(self, system: System) -> None:
        """Bind to a system."""
        if self.wrapped_input is not None:
            self.wrapped_input.bind(system)
            # Carry through dim if not set
            if getattr(self, 'dim', None) is None:
                self.dim = getattr(self.wrapped_input, 'dim', None)
        self._system = system
    
    def to(self, device: str):
        """
        Move grid cache to specified device.
        
        Args:
            device: Device to move to ('cpu', 'cuda', 'cuda:0', etc.)
        
        Returns:
            self (for method chaining)
        """
        device = torch.device(device)
        
        if self._grid_cache is not None:
            self._grid_cache = self._grid_cache.to(device)
        
        if self._state_grid_points is not None:
            self._state_grid_points = [p.to(device) for p in self._state_grid_points]
        if self._time_grid_points is not None:
            self._time_grid_points = self._time_grid_points.to(device)
        self._cache_device = device
        return self

    def input(self,
              state: torch.Tensor,
              time: float) -> torch.Tensor:
        """
        Query input values from the cached grid.

        If interpolate=True, performs nearest-neighbor lookup; otherwise requires exact grid points.

        Args:
            state: State tensor of shape [..., state_dim]
            time: Time value

        Returns:
            Input tensor of shape [..., input_dim] from the grid (or interpolated if enabled)

        Raises:
            RuntimeError: If GridInput not properly initialized or not bound to a system
            ValueError: If interpolate=False and query state/time are off-grid
        """
        if self._grid_cache is None or self._state_grid_points is None:
            raise RuntimeError("GridInput must be constructed with precomputed grid tensors before calling input()")
        if self._system is None:
            raise RuntimeError("GridInput must be bound to a system to determine state dimensionality")

        if self.interpolate:
            return self._lookup_nearest_grid(state, time)
        else:
            return self._lookup_exact_grid(state, time)
    
    def _lookup_nearest_grid(self,
                             state: torch.Tensor,
                             time: float
    ) -> torch.Tensor:
        """
        Nearest-neighbor lookup of input values from the cached grid.

        For time-invariant inputs, snaps only across state dimensions.
        For time-varying inputs, also snaps to the nearest time sample.

        Args:
            state: State tensor of shape [..., state_dim]
            time: Time value (ignored for time-invariant inputs)

        Returns:
            Input tensor of shape [..., input_dim]
        """
        device = self._cache_device
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, device=device)
        else:
            state = state.to(device)

        original_shape = state.shape[:-1]
        state_flat = state.reshape(-1, state.shape[-1])
        batch_size = state_flat.shape[0]

        # State indices via shared utility
        idx_state = _nearest_state_indices_util(self._state_grid_points, state_flat)
        grid_indices = [idx_state[:, d] for d in range(self._system.state_dim)]

        # Time dimension (only for time-varying inputs): choose nearest time
        if not self.time_invariant:
            t_values = torch.full((batch_size,), float(time), device=device, dtype=self._time_grid_points.dtype)
            time_indices = _nearest_time_index_util(self._time_grid_points, t_values)
            grid_indices.append(time_indices)

        # Advanced indexing into cache and reshape
        index_tuple = tuple(idx_tensor for idx_tensor in grid_indices)
        result = self._grid_cache[index_tuple]
        result_shape = list(original_shape) + [self.dim]
        return result.reshape(result_shape)
    
    def _lookup_exact_grid(self,
                           state: torch.Tensor,
                           time: float
    ) -> torch.Tensor:
        """
        Look up input values from the cached grid for exact grid states only.
        
        This method requires that the queried state (and time for time-varying inputs)
        exactly matches a grid point. An error is raised if off-grid queries are made.
        
        Fully vectorized for maximum performance.
        
        Args:
            state: State tensor of shape [..., state_dim]
            time: Time value (ignored for time-invariant inputs)
        
        Returns:
            Input tensor of shape [..., input_dim] from the grid
        
        Raises:
            ValueError: If state or time do not match grid points
        """
        # Ensure state is on the cache device
        device = self._cache_device
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, device=device)
        else:
            state = state.to(device)

        original_shape = state.shape[:-1]
        state_flat = state.reshape(-1, state.shape[-1])
        batch_size = state_flat.shape[0]

        # Exact state indices via shared utility (raises if off-grid beyond tol)
        tolerance = 1e-6
        idx_state = _exact_state_indices_util(self._state_grid_points, state_flat, tol=tolerance)
        grid_indices = [idx_state[:, d] for d in range(self._system.state_dim)]

        # Time dimension (only for time-varying inputs): require exact match within tol
        if not self.time_invariant:
            t_idx = _exact_axis_indices_util(
                self._time_grid_points,
                torch.tensor([float(time)], device=device, dtype=self._time_grid_points.dtype),
                tol=tolerance,
            )[0]
            grid_indices.append(t_idx.expand(batch_size))

        # Look up values from grid using advanced indexing and reshape
        index_tuple = tuple(idx_tensor for idx_tensor in grid_indices)
        result = self._grid_cache[index_tuple]
        result_shape = list(original_shape) + [self.dim]
        return result.reshape(result_shape)
