"""Monte Carlo value function estimation via simulation sampling.

Provides utilities for empirically estimating value functions by running
many simulations from a grid of initial states and computing failure statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import pickle
import torch
from tqdm import tqdm

from src.core.simulators import simulate_euler
from src.core.systems import System
from src.core.inputs import Input
from src.impl.inputs.standalone.common.uniform_random import UniformRandomInput

__all__ = ["SliceSpec", "MonteCarloConfig", "MonteCarloValueEstimator"]


@dataclass
class SliceSpec:
    vary_dims: Tuple[int, int]
    fixed: Dict[int, float]


@dataclass
class MonteCarloConfig:
    grid_resolution: Tuple[int, int]
    dt: float
    total_samples_per_state: int
    snapshot_samples: List[int]


def _build_slice_grid(system: System, slice_spec: SliceSpec, grid_resolution: Tuple[int, int]) -> Tuple[List[torch.Tensor], torch.Tensor, Tuple[int, int]]:
    """Build a 2D slice grid states.

    Returns:
      axes: list of coordinate vectors for the two varying dims [x_axis, y_axis]
      states: [nx*ny, D] tensor of initial states covering the slice grid
      shape2d: (nx, ny)
    """
    D = system.state_dim
    (i, j) = slice_spec.vary_dims
    nx, ny = grid_resolution
    limits = system.state_limits
    xi = torch.linspace(float(limits[0, i]), float(limits[1, i]), nx)
    yj = torch.linspace(float(limits[0, j]), float(limits[1, j]), ny)
    X, Y = torch.meshgrid(xi, yj, indexing='ij')
    grid = torch.zeros((nx * ny, D), dtype=torch.float32)
    # Set defaults to midpoints
    mids = (limits[0] + limits[1]) / 2.0
    grid[:] = mids
    # Apply fixed dims
    for k, v in (slice_spec.fixed or {}).items():
        grid[:, int(k)] = float(v)
    # Set varying dims from meshgrid
    grid[:, i] = X.reshape(-1)
    grid[:, j] = Y.reshape(-1)
    return [xi, yj], grid, (nx, ny)


def compute_monte_carlo_value(
    *,
    system: System,
    control: Input,
    slice_spec: SliceSpec,
    grid_resolution: Tuple[int, int],
    total_samples_per_state: int,
    snapshot_samples: List[int],
    dt: float,
    device: torch.device,
) -> Dict[str, torch.Tensor | List[torch.Tensor] | Dict[str, object]]:
    """Compute Monte Carlo under-approximate value over a 2D slice.

    Produces V_N(x) = min_{i<=N} min_t phi(x_t) for each grid state x, with snapshots.
    """
    # Build grid
    axes, states2d, shape2d = _build_slice_grid(system, slice_spec, grid_resolution)
    nx, ny = shape2d

    # Time discretization from system.horizon and dt
    H = float(getattr(system, 'time_horizon'))
    # Snap steps to an integer count while keeping exact horizon coverage
    steps = max(1, int(round(H / float(dt))))
    dt = H / steps

    # Bind control to system if not already
    control.set_type('control')
    control.bind(system)

    # Random inputs for disturbance and uncertainty
    dist = UniformRandomInput(); dist.set_type('disturbance'); dist.bind(system)
    unc = UniformRandomInput();  unc.set_type('uncertainty');  unc.bind(system)

    # Device
    states2d = states2d.to(device)
    use_gpu = (device.type == 'cuda')

    # Running min value per grid point
    V = torch.full((nx * ny,), float('inf'), dtype=torch.float32, device=device)

    # Prepare snapshot checkpoints (sorted unique and <= total)
    snaps = sorted({int(s) for s in snapshot_samples if int(s) > 0 and int(s) <= total_samples_per_state})
    snapshots: List[torch.Tensor] = []

    # Determine batch settings
    # We'll simulate K samples at a time by repeating the grid across sample dimension
    sys_batch = getattr(system, '_batch_size', None)
    # number of trajectories per simulate_euler call = len(states2d) * K <= sys_batch (if set)
    n_grid = states2d.shape[0]
    if sys_batch is None or sys_batch <= 0:
        # Simulate all grid points per call; choose sample_batch to keep memory reasonable
        # Heuristic: target ~100k trajectories per call if not constrained
        target = 100000
        sample_batch = max(1, target // max(1, n_grid))
    else:
        sample_batch = max(1, int(sys_batch) // max(1, n_grid))

    total_done = 0
    pbar = tqdm(total=total_samples_per_state, desc='Monte Carlo samples', disable=False)
    while total_done < total_samples_per_state:
        K = min(sample_batch, total_samples_per_state - total_done)
        # Repeat states K times along batch
        initial = states2d.repeat(K, 1)

        result = simulate_euler(
            system=system,
            control=control,
            disturbance=dist,
            uncertainty=unc,
            dt=dt,
            num_steps=steps,
            initial_state=initial,
            show_progress=True,
            leave_progress=False,
            device=device,
        )
        # states: [K*n_grid, steps+1, D]
        states = result.states  # on device
        # Evaluate failure function along time
        with torch.no_grad():
            sdf_time = system.failure_function(states, None)
            # Ensure [B, T] in case of extra singleton dim
            if sdf_time.ndim > 2:
                sdf_time = sdf_time.squeeze(-1)
            # Min over time per trajectory, then reshape into [K, n_grid]
            g = torch.min(sdf_time, dim=1).values  # [K*n_grid]
            g = g.reshape(K, n_grid)               # [K, n_grid]

            prev_total = total_done
            # Identify snapshots that fall within this batch window (prev_total, prev_total+K]
            batch_snaps: List[int] = []
            while snaps and snaps[0] <= prev_total + K:
                batch_snaps.append(snaps.pop(0))

            # If we need snapshots within this batch, compute prefix minima only up to the
            # maximum requested offset to avoid a full cummin over K rows.
            if batch_snaps:
                max_rel = max(n - prev_total for n in batch_snaps)  # 1-based
                prefix_vals, _ = torch.cummin(g[:max_rel], dim=0)  # [max_rel, n_grid]
                for n_snap in batch_snaps:
                    rel = n_snap - prev_total  # 1-based within batch
                    V_snap = torch.minimum(V, prefix_vals[rel - 1])
                    snapshots.append(V_snap.detach().clone().reshape(nx, ny).cpu())

            # After processing the batch, update V with min across all K samples
            g_batch_min = torch.min(g, dim=0).values  # [n_grid]
            V = torch.minimum(V, g_batch_min)

        total_done += K
        pbar.update(K)

    pbar.close()

    # Final value grid
    V_grid = V.reshape(nx, ny).detach().cpu()

    return {
        'axes': [a.cpu() for a in axes],
        'value': V_grid,              # [nx, ny]
        'snapshots': snapshots,       # list of [nx, ny]
        'meta': {
            'shape2d': [nx, ny],
            'vary_dims': list(slice_spec.vary_dims),
            'fixed': {int(k): float(v) for k, v in (slice_spec.fixed or {}).items()},
            'dt': float(dt),
            'steps': int(steps),
        }
    }


def save_monte_carlo_cache(
    *,
    tag: str,
    system: System,
    control_name: str,
    description: str,
    payload: Dict[str, object],
) -> None:
    """Save Monte Carlo value payload to .cache with metadata."""
    cache_dir = Path('.cache') / 'monte_carlo_values'
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{tag}.pkl'

    data = {
        'system_name': type(system).__name__,
        'control_name': control_name,
        'description': description,
        'time_horizon': float(getattr(system, 'time_horizon')),
        'state_labels': getattr(system, 'state_labels', tuple()),
        'axes': payload['axes'],
        'value': payload['value'],
        'snapshots': payload['snapshots'],
        'meta': payload['meta'],
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Minimal JSON metadata for fast inspection
    meta = {
        'system_name': data['system_name'],
        'control_name': data['control_name'],
        'description': data['description'],
        'shape2d': data['meta']['shape2d'],
        'vary_dims': data['meta']['vary_dims'],
        'snapshot_count': len(data['snapshots']),
        'time_horizon': data['time_horizon'],
    }
    with open(path.with_suffix('.meta.json'), 'w') as jf:
        json.dump(meta, jf, indent=2)
