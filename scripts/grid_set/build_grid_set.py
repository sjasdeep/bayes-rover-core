#!/usr/bin/env python3
"""
Build a tagged GridSet cache for a specific system and GridInput cache tag.

This script reuses the state/time grid and nominal input values from an existing
GridInput cache (by tag), computes uncertainty sets using the system's
uncertainty limits, and saves a GridSet payload at .cache/grid_sets/{TAG}.pkl.

Usage:
    # List available systems and GridInput tags
    python scripts/grid_set/build_grid_set.py --list

    # Build a new GridSet cache (errors if TAG exists)
    python scripts/grid_set/build_grid_set.py \
        --system RoverDark \
        --grid-input-tag {GRID_INPUT_TAG} \
        --tag {TAG} \
        --description "My GridSet" \
        --set-type box
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.cache_loaders import (
    get_grid_input_metadata,
    get_grid_input_payload,
    instantiate_system_by_name,
)
from src.utils.config import load_resolution_config
from src.utils.grids import compute_strides
from src.utils.registry import get_available_system_classes, list_grid_input_tags

# Module-level globals for worker processes (populated before forking workers)
_W_SHAPE = None
_W_AXES = None
_W_INPUT_DIM = None
_W_STATE_DIM = None
_W_GLOBAL_STRIDES = None
_W_NUM_CORNERS = None
_W_GRID_CACHE = None  # numpy view of grid_cache
_W_HAS_TIME_AXIS = None
_W_E_MIN = None  # numpy shared memory view [N, state_dim]
_W_E_MAX = None  # numpy shared memory view [N, state_dim]


def _flatten_multi_index_np(idxs: np.ndarray, strides: List[int]) -> np.ndarray:
    strides_arr = np.asarray(strides, dtype=np.int64)
    return (idxs.astype(np.int64, copy=False) * strides_arr).sum(axis=1).astype(np.int64, copy=False)


# ---- Shared NumPy helpers for worker functions ----
def _center_vals_from_multi_idx_np(multi_idx, axes: List[np.ndarray]) -> np.ndarray:
    return np.array([axes[d][multi_idx[d]] for d in range(len(axes))], dtype=np.float64)


def _compute_neighbor_ranges_np(center_vals: np.ndarray, e_min: np.ndarray, e_max: np.ndarray, axes: List[np.ndarray]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for d, ax in enumerate(axes):
        n = ax.shape[0]
        lo_v = center_vals[d] + e_min[d]
        hi_v = center_vals[d] + e_max[d]
        if lo_v > hi_v:
            lo_v, hi_v = hi_v, lo_v
        lo = int(np.searchsorted(ax, lo_v, side='left'))
        hi = int(np.searchsorted(ax, hi_v, side='right') - 1)
        if lo < 0:
            lo = 0
        elif lo >= n:
            lo = n - 1
        if hi < -1:
            hi = -1
        elif hi >= n:
            hi = n - 1
        if hi < lo:
            hi = lo
        ranges.append((lo, hi))
    return ranges


def _slice_local_vals_np(ranges: List[Tuple[int, int]], grid_slice: np.ndarray, input_dim: int) -> Tuple[np.ndarray, List[int]]:
    idx_tuple = tuple(slice(lo, hi + 1) for (lo, hi) in ranges) + (slice(None),)
    local_vals = grid_slice[idx_tuple].reshape(-1, input_dim)
    local_lengths = [r[1] - r[0] + 1 for r in ranges]
    return local_vals, local_lengths


def _local_to_global_flat_np(j_loc: int, local_lengths: List[int], ranges: List[Tuple[int, int]], global_strides: List[int]) -> int:
    if len(local_lengths) == 0:
        return 0
    offs = list(np.unravel_index(int(j_loc), tuple(local_lengths), order='C'))
    g_indices_arr = np.asarray([ranges[d][0] + offs[d] for d in range(len(local_lengths))], dtype=np.int64).reshape(1, -1)
    flat_idx_val = int(_flatten_multi_index_np(g_indices_arr, global_strides)[0])
    return flat_idx_val


# ---- Checkpoint helpers ----
def _ckpt_path_for(tag: str) -> Path:
    out_dir = Path('.cache') / 'grid_sets'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{tag}.ckpt.pkl"


def _save_checkpoint(
    *,
    tag: str,
    set_type: str,
    payload_meta: dict,
    progress_time_steps: int,
    box_lower: Optional[torch.Tensor] = None,
    box_upper: Optional[torch.Tensor] = None,
    box_state_est_corner_idx: Optional[torch.Tensor] = None,
    hull_vertices_padded: Optional[torch.Tensor] = None,
    hull_vertices_mask: Optional[torch.Tensor] = None,
    hull_state_idx_padded: Optional[torch.Tensor] = None,
) -> None:
    ckpt = {
        **payload_meta,
        'tag': tag,
        'set_type': set_type,
        'progress_time_steps': int(progress_time_steps),
    }
    if set_type == 'box':
        ckpt['box_lower'] = box_lower
        ckpt['box_upper'] = box_upper
        ckpt['box_state_est_corner_idx'] = box_state_est_corner_idx
    else:
        ckpt['box_lower'] = box_lower
        ckpt['box_upper'] = box_upper
        ckpt['hull_vertices_padded'] = hull_vertices_padded
        ckpt['hull_vertices_mask'] = hull_vertices_mask
        ckpt['hull_state_idx_padded'] = hull_state_idx_padded
    ckpt_path = _ckpt_path_for(tag)
    with open(ckpt_path, 'wb') as f:
        pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_checkpoint(tag: str) -> Optional[dict]:
    ckpt_path = _ckpt_path_for(tag)
    if not ckpt_path.exists():
        return None
    try:
        with open(ckpt_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


# ---- Hull padding helpers (shared by checkpoint and finalization) ----
def _pad_hulls_from_lists(hull_vertices: List[torch.Tensor], hull_state_idx: List[torch.Tensor], *, input_dim: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(hull_vertices) == 0:
        n_cells = 0
        hull_vertices_padded = torch.zeros((n_cells, 1, input_dim), dtype=dtype)
        hull_vertices_mask = torch.zeros((n_cells, 1), dtype=torch.bool)
        hull_state_idx_padded = torch.zeros((n_cells, 1), dtype=torch.int32)
        return hull_vertices_padded, hull_vertices_mask, hull_state_idx_padded
    vmax = max(int(v.shape[0]) for v in hull_vertices) if len(hull_vertices) > 0 else 0
    n_cells = len(hull_vertices)
    if vmax == 0:
        hull_vertices_padded = torch.zeros((n_cells, 1, input_dim), dtype=dtype)
        hull_vertices_mask = torch.zeros((n_cells, 1), dtype=torch.bool)
        hull_state_idx_padded = torch.zeros((n_cells, 1), dtype=torch.int32)
    else:
        hull_vertices_padded = torch.zeros((n_cells, vmax, input_dim), dtype=dtype)
        hull_vertices_mask = torch.zeros((n_cells, vmax), dtype=torch.bool)
        hull_state_idx_padded = torch.zeros((n_cells, vmax), dtype=torch.int32)
        for i, verts in enumerate(hull_vertices):
            nv = int(verts.shape[0])
            if nv > 0:
                hull_vertices_padded[i, :nv, :] = verts.to(dtype)
                hull_vertices_mask[i, :nv] = True
                if i < len(hull_state_idx) and hull_state_idx[i] is not None:
                    hull_state_idx_padded[i, :nv] = hull_state_idx[i]
    return hull_vertices_padded, hull_vertices_mask, hull_state_idx_padded


def _unpad_hulls_to_lists(hull_vertices_padded: torch.Tensor, hull_vertices_mask: torch.Tensor, hull_state_idx_padded: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    verts_list: List[torch.Tensor] = []
    idx_list: List[torch.Tensor] = []
    if hull_vertices_padded is None or hull_vertices_mask is None or hull_state_idx_padded is None:
        return verts_list, idx_list
    n_cells = int(hull_vertices_padded.shape[0])
    for i in range(n_cells):
        mask_row = hull_vertices_mask[i]
        if mask_row.any():
            nv = int(mask_row.sum().item())
            verts_list.append(hull_vertices_padded[i, :nv, :])
            idx_list.append(hull_state_idx_padded[i, :nv])
        else:
            verts_list.append(torch.zeros((0, hull_vertices_padded.shape[-1]), dtype=hull_vertices_padded.dtype))
            idx_list.append(torch.zeros((0,), dtype=torch.int32))
    return verts_list, idx_list


def _process_cell_box(args):
    """Worker function for a single state cell (box set only).

    Returns a tuple (l, u, corner_idxs) where:
      - l, u: numpy arrays of shape (input_dim,)
      - corner_idxs: numpy array of shape (2^input_dim,), dtype=int32 with global flat state indices
    """
    import numpy as _np
    i_flat, ti = args

    # Read from module-level globals
    shape = _W_SHAPE
    axes = _W_AXES
    input_dim = _W_INPUT_DIM
    state_dim = _W_STATE_DIM
    global_strides = _W_GLOBAL_STRIDES
    num_corners = _W_NUM_CORNERS
    # Time slice view
    grid_slice = _W_GRID_CACHE[..., ti, :] if _W_HAS_TIME_AXIS else _W_GRID_CACHE
    e_min_all = _W_E_MIN
    e_max_all = _W_E_MAX

    # unravel flat index to multi-index (row-major / C-order)
    multi_idx = list(_np.unravel_index(i_flat, shape, order='C'))

    # Center state values per dim and neighbor ranges
    center_vals = _center_vals_from_multi_idx_np(multi_idx, axes)
    e_min = e_min_all[i_flat]
    e_max = e_max_all[i_flat]
    ranges = _compute_neighbor_ranges_np(center_vals, e_min, e_max, axes)

    # Slice values in neighborhood
    local_vals, local_lengths = _slice_local_vals_np(ranges, grid_slice, input_dim)

    # Fallback if empty
    if local_vals.size == 0:
        center_val = grid_slice[tuple(multi_idx + [slice(None)])]
        local_vals = center_val.reshape(1, -1)

    # Box bounds
    l = local_vals.min(axis=0)
    u = local_vals.max(axis=0)

    # Corner provenance: nearest state indices for each corner vector
    corner_idxs = _np.empty((num_corners,), dtype=_np.int32)
    # Precompute local grid unraveling helper
    for c in range(num_corners):
        # Build corner vector using lexicographic order over dims (LSB corresponds to last input dim)
        bits = [(c >> k) & 1 for k in range(input_dim)]
        corner_vec = _np.array([u[d] if bits[d] == 1 else l[d] for d in range(input_dim)], dtype=local_vals.dtype).reshape(1, -1)
        diffs = ((local_vals - corner_vec) ** 2).sum(axis=1)
        j_loc = int(diffs.argmin())
        # Map local j to global flat index
        flat_idx_val = _local_to_global_flat_np(j_loc, local_lengths, ranges, global_strides)
        corner_idxs[c] = _np.int32(flat_idx_val)

    return l, u, corner_idxs


def _process_cell_hull(args):
    """Worker function for a single state cell (hull set).

    Returns a tuple (l, u, verts, verts_state) where:
      - l, u: numpy arrays of shape (input_dim,)
      - verts: numpy array [nv, input_dim]
      - verts_state: numpy array [nv, state_dim]
    """
    import numpy as _np
    from scipy.spatial import ConvexHull as _ConvexHull  # imported here for worker processes

    i_flat, ti = args
    # Read from module-level globals
    shape = _W_SHAPE
    axes = _W_AXES
    input_dim = _W_INPUT_DIM
    state_dim = _W_STATE_DIM
    global_strides = _W_GLOBAL_STRIDES
    grid_slice = _W_GRID_CACHE[..., ti, :] if _W_HAS_TIME_AXIS else _W_GRID_CACHE
    e_min_all = _W_E_MIN
    e_max_all = _W_E_MAX

    multi_idx = list(_np.unravel_index(i_flat, shape, order='C'))

    center_vals = _center_vals_from_multi_idx_np(multi_idx, axes)
    e_min = e_min_all[i_flat]
    e_max = e_max_all[i_flat]

    # Neighbor index ranges per dim (inclusive indices for values within [lo_v, hi_v])
    ranges = _compute_neighbor_ranges_np(center_vals, e_min, e_max, axes)

    # Slice values and states in neighborhood
    local_vals, local_lengths = _slice_local_vals_np(ranges, grid_slice, input_dim)

    # Build corresponding local states [*, state_dim]
    if len(ranges) > 0:
        axis_slices = [axes[d][ranges[d][0]:ranges[d][1] + 1] for d in range(state_dim)]
        grids = _np.meshgrid(*axis_slices, indexing='ij')
        local_states = _np.stack(grids, axis=-1).reshape(-1, state_dim).astype(_np.float32, copy=False)
    else:
        local_states = _np.zeros((local_vals.shape[0], 0), dtype=_np.float32)

    # Fallback if empty
    if local_vals.size == 0:
        center_val = grid_slice[tuple(multi_idx + [slice(None)])]
        local_vals = center_val.reshape(1, -1)
        center_state = _np.array([axes[d][multi_idx[d]] for d in range(state_dim)], dtype=_np.float32)
        local_states = center_state.reshape(1, -1)

    # Box bounds
    l = local_vals.min(axis=0)
    u = local_vals.max(axis=0)

    # Deduplicate and convex hull
    lv_np = local_vals
    ls_np = local_states
    loc_indices = _np.arange(lv_np.shape[0], dtype=_np.int64)  # position within local slice
    # Special-case 1D control: hull is just [min,max] endpoints
    if input_dim == 1:
        j_min = int(lv_np[:, 0].argmin())
        j_max = int(lv_np[:, 0].argmax())
        verts = _np.array([[l[0]], [u[0]]], dtype=_np.float32)
        sel_js = _np.array([j_min, j_max], dtype=_np.int64)
        # map local j -> global flat state index
        if len(local_lengths) == 0:
            g_flat = _np.zeros((2,), dtype=_np.int32)
        else:
            g_flat_list = []
            for j in sel_js.tolist():
                flat_idx_val = _local_to_global_flat_np(int(j), local_lengths, ranges, global_strides)
                g_flat_list.append(_np.int32(flat_idx_val))
            g_flat = _np.asarray(g_flat_list, dtype=_np.int32)
        return l, u, verts, g_flat
    else:
        if lv_np.shape[0] > 1:
            q = _np.round(lv_np, 8)
            _, unique_idx = _np.unique(q, axis=0, return_index=True)
            lv_np = lv_np[unique_idx]
            ls_np = ls_np[unique_idx]
            loc_indices = loc_indices[unique_idx]
        try:
            hull = _ConvexHull(lv_np, qhull_options='QJ') if lv_np.shape[0] > input_dim else None
            if hull is None:
                verts = lv_np.astype(_np.float32, copy=False)
                sel_js = loc_indices.astype(_np.int64, copy=False)
            else:
                verts = lv_np[hull.vertices].astype(_np.float32, copy=False)
                sel_js = loc_indices[hull.vertices].astype(_np.int64, copy=False)
        except Exception:
            # Fallback: use deduplicated points; avoid explosion by optionally capping vertices
            verts = lv_np.astype(_np.float32, copy=False)
            sel_js = loc_indices.astype(_np.int64, copy=False)

        # map selected local j -> global flat state indices
        if len(local_lengths) == 0 or verts.shape[0] == 0:
            g_flat = _np.zeros((verts.shape[0],), dtype=_np.int32)
        else:
            g_flat = _np.empty((verts.shape[0],), dtype=_np.int32)
            for ii, j_loc in enumerate(sel_js.tolist()):
                flat_idx_val = _local_to_global_flat_np(int(j_loc), local_lengths, ranges, global_strides)
                g_flat[ii] = _np.int32(flat_idx_val)

        return l, u, verts, g_flat


def list_available():
    """List available systems and GridInput cache tags."""
    systems = get_available_system_classes()
    print("\nSystems:")
    for cls in systems:
        print(f"  - {cls.__name__}")
    print("\nGridInput cache tags:")
    tags = list_grid_input_tags()
    if not tags:
        print("  (none)")
    for t in tags:
        print(f"  - {t}")


def _prepare_axes(state_grid_points: List[torch.Tensor]):
    axes = [ax.detach().cpu() for ax in state_grid_points]
    shape = tuple(len(ax) for ax in axes)
    return axes, shape


def _compute_uncertainty_bounds(system, states: torch.Tensor, time_val: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # Expect system.uncertainty_limits to accept batched states and scalar time
    e_min, e_max = system.uncertainty_limits(states, float(time_val))
    return e_min.detach().cpu(), e_max.detach().cpu()


def build_grid_set(system_name: str, grid_input_tag: str, set_type: str, tag: str, description: str = "",
                   *, config_path: str = 'config/resolutions.yaml',
                   time_resolution: Optional[int] = None, workers: Optional[int] = -1,
                   force: bool = False, checkpoint_every: int = 10) -> bool:
    # Validate system
    try:
        system = instantiate_system_by_name(system_name)
    except Exception as e:
        print(f"Error: System '{system_name}' not found or failed to instantiate: {e}")
        return False
    if set_type not in ('box', 'hull'):
        print("Error: --set-type must be 'box' or 'hull'")
        return False

    # Check destination
    out_dir = Path('.cache') / 'grid_sets'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.pkl"
    ckpt_path = _ckpt_path_for(tag)
    if out_path.exists():
        if not force:
            # If final artifact exists, assume build is complete.
            print(f"✓ GridSet cache already exists: {out_path}")
            return True
        else:
            print(f"\n⚠ Overwriting existing cache: {out_path.name}")
            out_path.unlink(missing_ok=True)
            ckpt_path.unlink(missing_ok=True)

    # Load GridInput metadata and payload
    meta = get_grid_input_metadata(grid_input_tag)
    gi_system = meta.get('system_name')
    input_name = meta.get('input_name')
    if gi_system != system_name:
        print(f"Error: GridInput tag '{grid_input_tag}' was built for system '{gi_system}', not '{system_name}'")
        if gi_system == 'RoverDark' and 'RoverDark' in system_name:
            print(f"Overriding system mismatch for '{system_name}' using GridInput from '{gi_system}'")
        else:
            return False
    gi = get_grid_input_payload(grid_input_tag)

    # We already instantiated 'system' above; consult its properties (e.g., time_invariant_uncertainty_limits)

    state_grid_points: List[torch.Tensor] = [t.detach().cpu() for t in gi['state_grid_points']]
    time_grid_points: torch.Tensor = gi.get('time_grid_points')
    if time_grid_points is not None:
        # GridInput had an explicit time axis; reuse it
        time_grid_points = time_grid_points.detach().cpu()
        print(f"Using time grid from GridInput: {time_grid_points.shape[0]} time steps")
    else:
        # GridInput is time-invariant. If the system declares that its uncertainty
        # limits are time-invariant, we keep GridSet time-invariant. Otherwise,
        # synthesize a time grid from config (or CLI overrides) to evaluate
        # time-varying bounds across time.
        is_time_invariant_limits = getattr(system, 'time_invariant_uncertainty_limits', False)
        if is_time_invariant_limits:
            time_grid_points = None
            print(f"GridSet will be time-invariant (System.time_invariant_uncertainty_limits={is_time_invariant_limits})")
        else:
            print(f"System has time-varying uncertainty limits (System.time_invariant_uncertainty_limits={is_time_invariant_limits})")
            cfg = load_resolution_config(system_name, input_name, config_path)
            tr = time_resolution if time_resolution is not None else (cfg.get('time_resolution') if cfg else None)
            th = float(getattr(system, 'time_horizon'))
            if tr is not None:
                time_grid_points = torch.linspace(0.0, th, int(tr))
                print(f"Synthesized time grid: {time_grid_points.shape[0]} steps over horizon {th}")
            else:
                time_grid_points = None  # remain time-invariant if no config available
                print(f"⚠ Warning: No time resolution available, GridSet will remain time-invariant despite time-varying limits")
    grid_cache: torch.Tensor = gi['grid_cache'].detach().cpu()  # [n1,...,nk,(nt), input_dim]

    state_dim = len(state_grid_points)
    axes, shape = _prepare_axes(state_grid_points)
    has_time_axis = (grid_cache.ndim == len(shape) + 2)
    t_size = 1 if time_grid_points is None else int(time_grid_points.numel())
    input_dim = int(grid_cache.shape[-1])

    # Prepare outputs (or resume from checkpoint)
    if set_type == 'box':
        box_lower = torch.empty(*shape, t_size, input_dim, dtype=grid_cache.dtype)
        box_upper = torch.empty_like(box_lower)
        hull_vertices = None
        # Provenance: per-corner estimated state flat indices (2^U corners)
        num_corners = (1 << input_dim)
        box_state_est_corner_idx = torch.empty(*shape, t_size, num_corners, dtype=torch.int32)
    else:
        # Hulls: store vertices list per grid point/time; also prepare box approximation tensors
        hull_vertices = []
        hull_state_idx = []  # per-vertex global flat state indices
        box_lower = torch.empty(*shape, t_size, input_dim, dtype=grid_cache.dtype)
        box_upper = torch.empty_like(box_lower)

    # Attempt resume from checkpoint
    start_ti = 0
    ckpt = _load_checkpoint(tag)
    if ckpt is not None:
        try:
            if ckpt.get('set_type') != set_type:
                print(f"⚠ Checkpoint set_type mismatch (ckpt={ckpt.get('set_type')}, arg={set_type}); ignoring checkpoint.")
            else:
                ckpt_shape = tuple(int(x) for x in ckpt.get('grid_shape', ()))
                if ckpt_shape and ckpt_shape != (shape + (t_size,)):
                    print(f"⚠ Checkpoint grid shape mismatch (ckpt={ckpt_shape}, expected={(shape + (t_size,))}); ignoring checkpoint.")
                else:
                    start_ti = int(ckpt.get('progress_time_steps', 0))
                    if set_type == 'box':
                        box_lower = ckpt['box_lower']
                        box_upper = ckpt['box_upper']
                        box_state_est_corner_idx = ckpt['box_state_est_corner_idx']
                        print(f"Resuming from checkpoint at time index {start_ti}/{t_size} (box)")
                    else:
                        box_lower = ckpt['box_lower']
                        box_upper = ckpt['box_upper']
                        hvp = ckpt.get('hull_vertices_padded')
                        hvm = ckpt.get('hull_vertices_mask')
                        hsp = ckpt.get('hull_state_idx_padded')
                        if hvp is not None and hvm is not None and hsp is not None:
                            hull_vertices, hull_state_idx = _unpad_hulls_to_lists(hvp, hvm, hsp)
                        else:
                            # Backward compatibility: older checkpoints stored lists
                            hull_vertices = ckpt.get('hull_vertices', [])
                            hull_state_idx = ckpt.get('hull_state_idx', [])
                        print(f"Resuming from checkpoint at time index {start_ti}/{t_size} (hull)")
        except Exception as e:
            print(f"⚠ Failed to load checkpoint, starting fresh: {e}")

    # Precompute global strides for state axes (row-major over state dims)
    state_sizes = list(shape)
    global_strides: List[int] = compute_strides(state_sizes)

    # Iterate times
    total_states = int(torch.tensor(shape).prod().item())
    print(f"Building GridSet: set_type={set_type}, states={total_states}, times={t_size}, input_dim={input_dim}")
    # Flatten multi-index iteration using nested loops per axis
    from tqdm import tqdm

    # Parallel processing setup: single persistent pool and shared memory for per-time arrays
    parallel_enabled = (set_type in ('box', 'hull') and (workers is not None and int(workers) != 1))
    executor = None
    shm_e_min = None
    shm_e_max = None
    try:
        import numpy as _np
        # Set worker globals (always), so we can reuse worker functions with workers=1
        axes_np = [a.numpy() for a in axes]
        global _W_SHAPE, _W_AXES, _W_INPUT_DIM, _W_STATE_DIM, _W_GLOBAL_STRIDES, _W_NUM_CORNERS, _W_GRID_CACHE, _W_HAS_TIME_AXIS, _W_E_MIN, _W_E_MAX
        _W_SHAPE = tuple(shape)
        _W_AXES = axes_np
        _W_INPUT_DIM = input_dim
        _W_STATE_DIM = state_dim
        _W_GLOBAL_STRIDES = list(global_strides)
        _W_NUM_CORNERS = (1 << input_dim)
        _W_HAS_TIME_AXIS = bool(has_time_axis)
        # Numpy view of grid_cache (read-only usage)
        _W_GRID_CACHE = grid_cache.numpy()

        if parallel_enabled:
            import os
            from multiprocessing import get_context, shared_memory

            # Allocate shared memory for e_min/e_max arrays once (N x state_dim)
            N = int(_np.prod(shape))
            sd = int(state_dim)
            # Prefer float32 shared buffers unless grid_cache is float64
            if grid_cache.dtype == torch.float64:
                np_dtype = _np.float64
            else:
                np_dtype = _np.float32
            shm_e_min = shared_memory.SharedMemory(create=True, size=N * sd * _np.dtype(np_dtype).itemsize)
            shm_e_max = shared_memory.SharedMemory(create=True, size=N * sd * _np.dtype(np_dtype).itemsize)
            _W_E_MIN = _np.ndarray((N, sd), dtype=np_dtype, buffer=shm_e_min.buf)
            _W_E_MAX = _np.ndarray((N, sd), dtype=np_dtype, buffer=shm_e_max.buf)

            # Create persistent process pool with fork context
            w = int(workers) if workers is not None else -1
            maxw = max(1, (os.cpu_count() or 2) - 1) if w == -1 else w
            print(f"Using parallel processing with {maxw} worker processes (workers={workers})")
            mp_ctx = get_context('fork')
            from concurrent.futures import ProcessPoolExecutor
            executor = ProcessPoolExecutor(max_workers=maxw, mp_context=mp_ctx)

        # Time iteration progress bar
        for ti in tqdm(range(start_ti, t_size), desc="Time steps", leave=True):
            t_val = 0.0 if time_grid_points is None else float(time_grid_points[ti].item())
            # Build full state list for uncertainty in a batch
            meshes = torch.meshgrid(*axes, indexing='ij')
            grid_states = torch.stack(meshes, dim=-1).reshape(-1, state_dim)
            e_min_all, e_max_all = _compute_uncertainty_bounds(system, grid_states, t_val)

            if parallel_enabled:
                # Copy into shared memory buffers for this time step
                _np.copyto(_W_E_MIN, e_min_all.numpy().astype(_W_E_MIN.dtype, copy=False))
                _np.copyto(_W_E_MAX, e_max_all.numpy().astype(_W_E_MAX.dtype, copy=False))
                # Map over all states, passing (i_flat, ti)
                chunksize = max(1, total_states // (maxw * 8))
                if set_type == 'box':
                    for i_flat, (l_np, u_np, corner_np) in enumerate(tqdm(executor.map(_process_cell_box, ((i, ti) for i in range(total_states)), chunksize=chunksize), total=total_states, desc=f" time {ti+1}/{t_size}", leave=False)):
                        multi_idx = _np.unravel_index(i_flat, shape, order='C')
                        box_lower[tuple(list(multi_idx) + [ti, slice(None)])] = torch.from_numpy(l_np).to(grid_cache.dtype)
                        box_upper[tuple(list(multi_idx) + [ti, slice(None)])] = torch.from_numpy(u_np).to(grid_cache.dtype)
                        box_state_est_corner_idx[tuple(list(multi_idx) + [ti, slice(None)])] = torch.from_numpy(corner_np.astype(np.int32))
                else:
                    for i_flat, (l_np, u_np, verts_np, vidx_np) in enumerate(tqdm(executor.map(_process_cell_hull, ((i, ti) for i in range(total_states)), chunksize=chunksize), total=total_states, desc=f" time {ti+1}/{t_size}", leave=False)):
                        multi_idx = _np.unravel_index(i_flat, shape, order='C')
                        box_lower[tuple(list(multi_idx) + [ti, slice(None)])] = torch.from_numpy(l_np).to(grid_cache.dtype)
                        box_upper[tuple(list(multi_idx) + [ti, slice(None)])] = torch.from_numpy(u_np).to(grid_cache.dtype)
                        hull_vertices.append(torch.from_numpy(verts_np).to(grid_cache.dtype))
                        hull_state_idx.append(torch.from_numpy(vidx_np.astype(np.int32)))
                # Save checkpoint in parallel path as well
                if checkpoint_every and int(checkpoint_every) > 0:
                    do_ckpt = (((ti + 1) % int(checkpoint_every)) == 0) or (ti == t_size - 1)
                    if do_ckpt:
                        payload_meta = {
                            'description': description,
                            'system_name': system_name,
                            'input_name': input_name,
                            'grid_input_tag': grid_input_tag,
                            'grid_shape': shape + (t_size,),
                            'state_grid_points': state_grid_points,
                            'time_grid_points': time_grid_points,
                        }
                        if set_type == 'box':
                            _save_checkpoint(
                                tag=tag,
                                set_type=set_type,
                                payload_meta=payload_meta,
                                progress_time_steps=ti + 1,
                                box_lower=box_lower,
                                box_upper=box_upper,
                                box_state_est_corner_idx=box_state_est_corner_idx,
                            )
                        else:
                            hvp, hvm, hsp = _pad_hulls_from_lists(hull_vertices, hull_state_idx, input_dim=input_dim, dtype=box_lower.dtype)
                            _save_checkpoint(
                                tag=tag,
                                set_type=set_type,
                                payload_meta=payload_meta,
                                progress_time_steps=ti + 1,
                                box_lower=box_lower,
                                box_upper=box_upper,
                                hull_vertices_padded=hvp,
                                hull_vertices_mask=hvm,
                                hull_state_idx_padded=hsp,
                            )
                        print(f"⏺  Checkpoint saved at t={ti+1}/{t_size}: {ckpt_path}")
                continue  # finished this time slice in parallel mode

            # Unified sequential path using worker functions in-process
            if ti == start_ti:
                print(f"Using sequential processing (workers={workers})")
            # Provide E arrays directly via globals (no shared memory)
            if grid_cache.dtype == torch.float64:
                np_dtype = _np.float64
            else:
                np_dtype = _np.float32
            _W_E_MIN = e_min_all.numpy().astype(np_dtype, copy=False)
            _W_E_MAX = e_max_all.numpy().astype(np_dtype, copy=False)
            # Iterate locally and invoke the same worker logic
            if set_type == 'box':
                for i_flat in tqdm(range(total_states), total=total_states, desc=f" time {ti+1}/{t_size}", leave=False):
                    l_np, u_np, corner_np = _process_cell_box((i_flat, ti))
                    multi_idx = _np.unravel_index(i_flat, shape, order='C')
                    box_lower[tuple(list(multi_idx) + [ti, slice(None)])] = torch.from_numpy(l_np).to(grid_cache.dtype)
                    box_upper[tuple(list(multi_idx) + [ti, slice(None)])] = torch.from_numpy(u_np).to(grid_cache.dtype)
                    box_state_est_corner_idx[tuple(list(multi_idx) + [ti, slice(None)])] = torch.from_numpy(corner_np.astype(np.int32))
            else:
                for i_flat in tqdm(range(total_states), total=total_states, desc=f" time {ti+1}/{t_size}", leave=False):
                    l_np, u_np, verts_np, vidx_np = _process_cell_hull((i_flat, ti))
                    multi_idx = _np.unravel_index(i_flat, shape, order='C')
                    box_lower[tuple(list(multi_idx) + [ti, slice(None)])] = torch.from_numpy(l_np).to(grid_cache.dtype)
                    box_upper[tuple(list(multi_idx) + [ti, slice(None)])] = torch.from_numpy(u_np).to(grid_cache.dtype)
                    hull_vertices.append(torch.from_numpy(verts_np).to(grid_cache.dtype))
                    hull_state_idx.append(torch.from_numpy(vidx_np.astype(np.int32)))

            # Save checkpoint periodically
            if checkpoint_every and int(checkpoint_every) > 0:
                do_ckpt = (((ti + 1) % int(checkpoint_every)) == 0) or (ti == t_size - 1)
                if do_ckpt:
                    payload_meta = {
                        'description': description,
                        'system_name': system_name,
                        'input_name': input_name,
                        'grid_input_tag': grid_input_tag,
                        'grid_shape': shape + (t_size,),
                        'state_grid_points': state_grid_points,
                        'time_grid_points': time_grid_points,
                    }
                    if set_type == 'box':
                        _save_checkpoint(
                            tag=tag,
                            set_type=set_type,
                            payload_meta=payload_meta,
                            progress_time_steps=ti + 1,
                            box_lower=box_lower,
                            box_upper=box_upper,
                            box_state_est_corner_idx=box_state_est_corner_idx,
                        )
                    else:
                        # Pad current hull lists for faster checkpointing
                        hvp, hvm, hsp = _pad_hulls_from_lists(hull_vertices, hull_state_idx, input_dim=input_dim, dtype=box_lower.dtype)
                        _save_checkpoint(
                            tag=tag,
                            set_type=set_type,
                            payload_meta=payload_meta,
                            progress_time_steps=ti + 1,
                            box_lower=box_lower,
                            box_upper=box_upper,
                            hull_vertices_padded=hvp,
                            hull_vertices_mask=hvm,
                            hull_state_idx_padded=hsp,
                        )
                    print(f"⏺  Checkpoint saved at t={ti+1}/{t_size}: {ckpt_path}")
    finally:
        # Cleanup shared memory and executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        if shm_e_min is not None:
            try:
                shm_e_min.close()
                shm_e_min.unlink()
            except Exception:
                pass
        if shm_e_max is not None:
            try:
                shm_e_max.close()
                shm_e_max.unlink()
            except Exception:
                pass
    
    # If hull: pad vertices and state indices to speed up serialization and loading
    if set_type == 'hull':
        n_cells = len(hull_vertices)
        vmax = max((int(v.shape[0]) for v in hull_vertices), default=0)
        # Instrumentation: report hull vertex stats to spot blow-ups early
        try:
            total_nv = sum(int(v.shape[0]) for v in hull_vertices)
            avg_nv = (total_nv / n_cells) if n_cells > 0 else 0.0
            # Rough memory estimate for padded arrays: float32 (4 bytes), bool mask ~1 byte, indices int32 (4 bytes)
            bytes_vertices = n_cells * vmax * input_dim * 4
            bytes_indices = n_cells * vmax * 4
            bytes_mask = n_cells * vmax * 1
            est_mb = (bytes_vertices + bytes_indices + bytes_mask) / (1024 * 1024)
            print(f"Hull padding stats: cells={n_cells}, vmax={vmax}, avg_nv={avg_nv:.2f}, est_mem≈{est_mb:.1f} MB")
        except Exception:
            pass
        hull_vertices_padded, hull_vertices_mask, hull_state_idx_padded = _pad_hulls_from_lists(
            hull_vertices, hull_state_idx, input_dim=input_dim, dtype=box_lower.dtype
        )

    # Assemble payload
    payload = {
        'tag': tag,
        'description': description,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'system_name': system_name,
        'input_name': input_name,
        'set_type': set_type,
        'grid_input_tag': grid_input_tag,
        'grid_shape': shape + (t_size,),
        'state_grid_points': state_grid_points,
        'time_grid_points': time_grid_points,
    }
    if set_type == 'hull':
        payload['hull_vertices_padded'] = hull_vertices_padded
        payload['hull_vertices_mask'] = hull_vertices_mask
        payload['hull_state_idx_padded'] = hull_state_idx_padded
    else:
        payload['box_lower'] = box_lower
        payload['box_upper'] = box_upper
        payload['box_state_est_corner_idx'] = box_state_est_corner_idx

    # Timed serialization to help diagnose potential slowdowns
    import time as _time
    tmp_path = out_path.with_suffix('.pkl.tmp')
    print("Serializing GridSet payload to disk...")
    t0 = _time.time()
    with open(tmp_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
    t1 = _time.time()
    tmp_path.replace(out_path)
    t2 = _time.time()
    print(f"✓ Saved GridSet cache: {out_path} (serialize {t1 - t0:.2f}s, move {t2 - t1:.2f}s)")
    # Cleanup checkpoint on success
    try:
        _ckpt_path_for(tag).unlink(missing_ok=True)
    except Exception:
        pass
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config-driven usage:
  Resolution parameters come from config/resolutions.yaml.
  Use --set KEY=VALUE for ad-hoc overrides.

Examples:
  python build_grid_set.py --system RoverDark --grid-input-tag RoverDark_MPC --tag RoverDark_MPC_Box
  python build_grid_set.py --system RoverDark --grid-input-tag RoverDark_MPC --tag my_set --set set_type=hull
""",
    )
    parser.add_argument('--list', action='store_true', help='List systems and GridInput tags')
    parser.add_argument('--system', type=str, help='System class name')
    parser.add_argument('--grid-input-tag', type=str, help='Existing GridInput cache tag to reuse')
    parser.add_argument('--tag', type=str, help='New GridSet tag (filename without extension)')
    parser.add_argument('--config', type=str, default='config/resolutions.yaml', help='Resolution config for time grid')
    parser.add_argument('--force', action='store_true', help='Overwrite existing cache if it exists')
    
    # Generic override mechanism
    parser.add_argument('--set', type=str, action='append', dest='overrides', metavar='KEY=VALUE',
                        help='Override config value (can be repeated)')
    
    # Convenience args (shortcuts for --set)
    parser.add_argument('--description', type=str, default='', help='Description (convenience for --set description=X)')
    parser.add_argument('--set-type', type=str, choices=['box', 'hull'], default='box', help='Set representation (convenience for --set set_type=X)')
    parser.add_argument('--time-resolution', type=int, default=None, help='Override time resolution (convenience for --set time_resolution=X)')
    parser.add_argument('--workers', type=int, default=-1, help='Worker processes (convenience for --set workers=X)')
    parser.add_argument('--checkpoint-every', type=int, default=10, help='Checkpoint frequency (convenience for --set checkpoint_every=X)')
    
    args = parser.parse_args()
    
    if args.list:
        list_available()
        return
    if not args.system or not args.grid_input_tag or not args.tag:
        print("Error: --system, --grid-input-tag and --tag are required")
        sys.exit(1)
    
    # Build config from defaults and overrides
    from src.utils.config import parse_key_value_overrides, apply_overrides
    cfg = {
        'description': args.description,
        'set_type': args.set_type,
        'time_resolution': args.time_resolution,
        'workers': args.workers,
        'checkpoint_every': args.checkpoint_every,
    }
    
    # Apply --set overrides (highest priority)
    if args.overrides:
        set_overrides = parse_key_value_overrides(args.overrides)
        cfg = apply_overrides(cfg, set_overrides)
        print(f"Applied --set overrides: {set_overrides}")
    
    success = build_grid_set(
        args.system,
        args.grid_input_tag,
        cfg.get('set_type', 'box'),
        args.tag,
        cfg.get('description', ''),
        config_path=args.config,
        time_resolution=cfg.get('time_resolution'),
        workers=cfg.get('workers', -1),
        force=args.force,
        checkpoint_every=cfg.get('checkpoint_every', 10),
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
