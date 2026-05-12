"""General grid indexing utilities shared across GridInput, GridSet, and GridValue.

This module centralizes common patterns:
- Nearest/exact index lookup along 1D axes
- Vectorized state index lookup across multiple axes
- Time axis nearest index
- Stride computation and flattening of multi-indices
- Snapping fixed dimension values to nearest grid points
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import torch

Tensor = torch.Tensor


# ---------- Axis index helpers ----------

def nearest_axis_indices(axis: Tensor, values: Tensor) -> Tensor:
    """Nearest indices on a 1D sorted axis for given values.
    
    NOTE: This function uses torch.searchsorted and assumes the axis is sorted
    in ascending order. For descending axes, results will be incorrect.
    Use torch.argmin(torch.abs(axis - value)) for non-ascending axes.

    axis: [N] - assumed to be sorted in ascending order
    values: [B]
    returns: [B] (int64)
    """
    # Ensure values is contiguous for efficient searchsorted
    values = values.contiguous()
    inds = torch.searchsorted(axis, values, right=False)
    inds = torch.clamp(inds, 0, axis.numel() - 1)
    has_prev = inds > 0
    prev = torch.where(has_prev, inds - 1, inds)
    d_curr = torch.abs(values - axis[inds])
    d_prev = torch.abs(values - axis[prev])
    inds = torch.where(d_prev < d_curr, prev, inds)
    return inds.to(torch.int64)


def exact_axis_indices(axis: Tensor, values: Tensor, tol: float = 1e-6) -> Tensor:
    """Exact indices on a 1D axis for values; errors if off-grid by > tol.

    axis: [N]
    values: [B]
    returns: [B] (int64)
    """
    # Ensure values is contiguous for efficient searchsorted
    values = values.contiguous()
    inds = torch.searchsorted(axis, values, right=False)
    inds = torch.clamp(inds, 0, axis.numel() - 1)
    has_prev = inds > 0
    prev = torch.where(has_prev, inds - 1, inds)
    d_curr = torch.abs(values - axis[inds])
    d_prev = torch.abs(values - axis[prev])
    use_prev = d_prev < d_curr
    inds = torch.where(use_prev, prev, inds)
    dists = torch.where(use_prev, d_prev, d_curr)
    if torch.any(dists > tol):
        bi = int(torch.argmax(dists).item())
        raise ValueError(
            f"Value not on grid (max deviation {float(dists[bi].item()):.3e}). "
            f"Enable interpolation or snap to nearest."
        )
    return inds.to(torch.int64)


# ---------- State/time indexing ----------

def nearest_state_indices(axes: List[Tensor], states: Tensor) -> Tensor:
    """Vectorized nearest per axis for batched states.

    axes: list of [Ni]
    states: [B, D]
    returns: [B, D] indices
    """
    idxs = []
    for d, ax in enumerate(axes):
        idxs.append(nearest_axis_indices(ax, states[:, d]))
    return torch.stack(idxs, dim=1)


def exact_state_indices(axes: List[Tensor], states: Tensor, tol: float = 1e-6) -> Tensor:
    idxs = []
    for d, ax in enumerate(axes):
        idxs.append(exact_axis_indices(ax, states[:, d], tol))
    return torch.stack(idxs, dim=1)


def nearest_time_index(times: Tensor, t_values: Tensor | float) -> Tensor:
    """Nearest indices on a time axis for scalar or batched times.
    
    NOTE: This function assumes times are sorted in ascending order.

    times: [T] - assumed to be sorted in ascending order
    t_values: float or [B]
    returns: [B] int64
    """
    if isinstance(t_values, (float, int)):
        vals = torch.tensor([float(t_values)], dtype=times.dtype, device=times.device)
    elif isinstance(t_values, torch.Tensor):
        vals = t_values.reshape(-1).to(dtype=times.dtype, device=times.device)
    else:
        # Fallback: convert via torch.tensor
        vals = torch.tensor(t_values).reshape(-1).to(dtype=times.dtype, device=times.device)
    return nearest_axis_indices(times, vals)


# ---------- Strides and flattening ----------

def compute_strides(sizes: List[int]) -> List[int]:
    """Row-major strides for sizes.

    Example: sizes=[n0,n1,n2] -> strides=[n1*n2, n2, 1]
    """
    D = len(sizes)
    strides = [1] * D
    for i in range(D - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return strides


def flatten_multi_index(idxs: Tensor, strides: List[int]) -> Tensor:
    """Flatten [B,D] indices using provided row-major strides into [B]."""
    if idxs.numel() == 0:
        return torch.zeros((idxs.shape[0],), dtype=torch.int64)
    s = torch.as_tensor(strides, dtype=torch.int64, device=idxs.device)
    return torch.sum(idxs.to(torch.int64) * s, dim=1)


# ---------- Fixed dim snapping ----------

def snap_fixed_dims_to_axes(fixed: Dict[int, float], axes: List[Tensor]) -> Dict[int, float]:
    """Snap provided fixed dim values to nearest values on corresponding axes.

    Returns a new dict with snapped float values.
    """
    snapped: Dict[int, float] = {}
    for d, v in (fixed or {}).items():
        ax = axes[d]
        idx = int(torch.argmin(torch.abs(ax - float(v))).item())
        snapped[d] = float(ax[idx].item())
    return snapped


# ---------- NumPy helpers (for worker contexts) ----------

__all__ = [
    "nearest_axis_indices",
    "exact_axis_indices",
    "nearest_state_indices",
    "exact_state_indices",
    "nearest_time_index",
    "compute_strides",
    "flatten_multi_index",
    "snap_fixed_dims_to_axes",
]
