"""Worst-case uncertainty for RoverLight using RoverDark GridValue with time=s.

This input provides an uncertainty vector e of dimension 4 for RoverLight:
  - For (x, y, theta): uses optimal uncertainty from a RoverDark GridValue
    evaluated at time = s (the fourth state), i.e., e3 = argmax_d V dynamics.
  - For s: returns 0 (no uncertainty on the clock state).

Performance optimization: The optimal uncertainty is precomputed for all grid points
and all times, then cached to disk. At runtime, we just do a fast grid lookup.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from src.core.inputs import Input
from src.impl.systems.rover_light import RoverLight
from src.utils.cache_loaders import load_grid_value_by_tag

__all__ = ["RoverLight_WorstCaseUncertaintyFromValue"]


def _build_optimal_uncertainty_cache(grid_value, tag: str) -> dict:
    """Precompute optimal uncertainty for all grid points and all times."""
    print(f"Building optimal uncertainty cache for GridValue '{tag}'...")
    
    axes = grid_value._axes
    times = grid_value._times
    grid_shape = grid_value.grid_shape
    state_dim = len(axes)
    n_times = len(times)
    
    # Create meshgrid of all state grid points
    grids = torch.meshgrid(*axes, indexing='ij')
    flat_states = torch.stack([g.reshape(-1) for g in grids], dim=1)  # [N_grid, state_dim]
    n_grid = flat_states.shape[0]
    
    # Precompute optimal uncertainty for each (state, time) pair
    # Shape: [n_grid, n_times, state_dim]
    opt_unc_grid = torch.zeros((n_grid, n_times, state_dim), dtype=torch.float32)
    
    # Process in batches over grid points, one time at a time
    batch_size = 10000
    for t_idx in tqdm(range(n_times), desc="Computing optimal uncertainty grid"):
        t_val = times[t_idx]
        for start in range(0, n_grid, batch_size):
            end = min(start + batch_size, n_grid)
            states_batch = flat_states[start:end]
            
            # Get optimal uncertainty for this batch at this time
            opt_u = grid_value.optimal_uncertainty(states_batch, t_val, interpolate=False)
            opt_unc_grid[start:end, t_idx] = opt_u.to(torch.float32)
    
    # Reshape to [*grid_shape, n_times, state_dim]
    opt_unc_grid = opt_unc_grid.reshape(*grid_shape, n_times, state_dim)
    
    cache = {
        'tag': tag,
        'grid_shape': grid_shape,
        'n_times': n_times,
        'axes': [ax.clone() for ax in axes],
        'times': times.clone(),
        'optimal_uncertainty': opt_unc_grid,  # [*grid_shape, n_times, state_dim]
    }
    
    print(f"  Grid shape: {grid_shape}, {n_times} time steps")
    print(f"  Cache size: {opt_unc_grid.numel() * 4 / 1e6:.1f} MB")
    
    return cache


def _load_or_build_optimal_uncertainty_cache(grid_value, tag: str) -> dict:
    """Load optimal uncertainty cache from disk, or build and save if not present."""
    cache_dir = Path(".cache") / "custom" / "rover_light" / "optimal_uncertainty"
    cache_path = cache_dir / f"{tag}.pkl"
    
    if cache_path.exists():
        print(f"Loading optimal uncertainty cache from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Build cache
    cache = _build_optimal_uncertainty_cache(grid_value, tag)
    
    # Save to disk
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved optimal uncertainty cache to {cache_path}")
    
    return cache


def _lookup_optimal_uncertainty_from_cache(
    cache: dict,
    states: torch.Tensor,  # [N, state_dim]
    times: torch.Tensor,   # [N] - time for each state
) -> torch.Tensor:
    """Look up optimal uncertainty for given states and times using nearest-neighbor.
    
    Args:
        cache: optimal uncertainty cache dict
        states: [N, state_dim] tensor of states
        times: [N] tensor of times (one per state)
    
    Returns:
        [N, state_dim] tensor of optimal uncertainty vectors
    """
    from src.utils.grids import nearest_state_indices, nearest_time_index
    
    axes = cache['axes']
    cache_times = cache['times']
    opt_unc = cache['optimal_uncertainty']  # [*grid_shape, n_times, state_dim]
    grid_shape = cache['grid_shape']
    state_dim = len(grid_shape)
    n_times = cache['n_times']
    
    N = states.shape[0]
    
    # Get nearest grid indices for each state
    idx_state = nearest_state_indices(axes, states)  # [N, state_dim]
    
    # Get nearest time index for each time value
    idx_time = nearest_time_index(cache_times, times)  # [N]
    
    # Flatten the grid for indexing
    # opt_unc shape: [*grid_shape, n_times, state_dim]
    # We need to index with (state_idx, time_idx) for each sample
    
    # Compute flat state index
    strides = [1]
    for dim in reversed(grid_shape[1:]):
        strides.insert(0, strides[0] * dim)
    strides = torch.tensor(strides, dtype=torch.int64)
    flat_state_idx = (idx_state * strides).sum(dim=1)  # [N]
    
    # Reshape opt_unc to [n_grid, n_times, state_dim]
    n_grid = int(np.prod(grid_shape))
    opt_unc_flat = opt_unc.reshape(n_grid, n_times, state_dim)
    
    # Index: [N, state_dim]
    result = opt_unc_flat[flat_state_idx, idx_time]
    
    return result


class RoverLight_WorstCaseUncertaintyFromValue(Input):
    type = 'uncertainty'
    system_class = RoverLight
    dim = 4
    time_invariant = False

    def __init__(self, *, grid_value_tag: Optional[str] = None, interpolate: bool = True) -> None:
        self._grid_tag = (str(grid_value_tag) if grid_value_tag is not None else None)
        self._interpolate = bool(interpolate)
        self._system: Optional[RoverLight] = None
        self._grid_value = None
        self._opt_unc_cache = None

    def bind(self, system: RoverLight) -> None:
        if not isinstance(system, RoverLight):
            raise TypeError(
                f"{type(self).__name__} requires RoverLight system, got {type(system).__name__}"
            )
        self._system = system

        grid_tag = self._grid_tag
        if grid_tag is None:
            # Try simulation config fallback
            try:
                from src.utils.config import load_simulation_config
                cfg = load_simulation_config(type(system).__name__, type(self).__name__)
                grid_tag = cfg.get('uncertainty_grid_value_tag')
            except Exception:
                grid_tag = None
        if not grid_tag:
            raise ValueError(
                "GridValue tag is required. Provide grid_value_tag in constructor or set "
                "config/simulations.yaml under this input with key 'grid_value_tag'."
            )

        gv = load_grid_value_by_tag(grid_tag, interpolate=self._interpolate)
        self._grid_value = gv
        
        # Load or build optimal uncertainty cache
        self._opt_unc_cache = _load_or_build_optimal_uncertainty_cache(gv, grid_tag)

    def to(self, device: torch.device | str):  # pragma: no cover
        return self

    def input(self, state: torch.Tensor, time: float) -> torch.Tensor:
        if self._system is None or self._opt_unc_cache is None:
            raise RuntimeError("bind() must be called before using the uncertainty input.")

        state = torch.as_tensor(state)
        orig_dtype = state.dtype
        orig_device = state.device

        x3 = state[..., :3]
        s = state[..., 3]
        # Flatten for cache lookup
        x3_cpu = x3.detach().cpu().to(torch.float32).reshape(-1, 3)
        s_cpu = s.detach().cpu().to(torch.float32).reshape(-1)

        # Fast cached lookup for optimal uncertainty
        e3 = _lookup_optimal_uncertainty_from_cache(self._opt_unc_cache, x3_cpu, s_cpu)
        
        # Append zero for s-dim
        zeros = torch.zeros((e3.shape[0], 1), dtype=e3.dtype)
        e4 = torch.cat((e3, zeros), dim=-1)
        # Reshape back to batch and move to original device/dtype
        e4 = e4.reshape(*state.shape[:-1], 4).to(dtype=orig_dtype, device=orig_device)
        return e4
