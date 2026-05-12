#!/usr/bin/env python3
"""
Clamp a cached GridSet's box bounds to the system's control limits and save as a new tag.

Usage:
  python scripts/grid_set/constrain_grid_set.py \
      --grid-set-tag RoverDark_MPC_NN_Box \
      --tag RoverDark_MPC_NN_Box_clamped \
      --description "Clamped to system control limits"
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.cache_loaders import (
    get_grid_set_payload,
    instantiate_system_by_name,
)


def _flatten_state_mesh(state_axes: list[torch.Tensor]) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    axes_cpu = [ax.detach().cpu() for ax in state_axes]
    meshes = torch.meshgrid(*axes_cpu, indexing='ij')
    grid_states = torch.stack(meshes, dim=-1).reshape(-1, len(axes_cpu))
    shape = tuple(len(ax) for ax in axes_cpu)
    return grid_states, shape


def clamp_grid_set_to_limits(grid_set_tag: str, out_tag: str, description: str = "", *, force: bool = False) -> bool:
    # Load source payload (raw dict) to preserve metadata fields
    src = get_grid_set_payload(grid_set_tag)

    if src.get('set_type') != 'box':
        print(f"Error: Only 'box' GridSet is supported (found set_type={src.get('set_type')})")
        return False

    system_name = src['system_name']
    system = instantiate_system_by_name(system_name)

    state_axes = src['state_grid_points']
    time_axis = src.get('time_grid_points')
    box_lower: torch.Tensor = src['box_lower']
    box_upper: torch.Tensor = src['box_upper']
    input_dim = int(box_lower.shape[-1])

    # Build flattened state list once
    flat_states, shape = _flatten_state_mesh(state_axes)
    N = int(flat_states.shape[0])
    T = 1 if time_axis is None else int(time_axis.numel())

    # Prepare output tensors
    dtype = box_lower.dtype
    device_cpu = torch.device('cpu')
    out_lower = torch.empty((*shape, T, input_dim), dtype=dtype)
    out_upper = torch.empty_like(out_lower)

    # Iterate time slices and clamp elementwise using system.control_limits
    for ti in range(T):
        t_val = 0.0 if time_axis is None else float(time_axis[ti].item())
        ctrl_lo, ctrl_hi = system.control_limits(flat_states, t_val)  # [N, U]
        # Reshape control limits to grid for broadcasting with per-cell bounds
        ctrl_lo_grid = ctrl_lo.reshape(*shape, input_dim)
        ctrl_hi_grid = ctrl_hi.reshape(*shape, input_dim)

        # View the source slice in grid shape
        src_lo_grid = box_lower[..., ti, :]
        src_hi_grid = box_upper[..., ti, :]

        # Clamp per element: lower = clamp(lower, ctrl_lo, ctrl_hi); upper likewise
        lo_clamped = torch.maximum(src_lo_grid, ctrl_lo_grid)
        lo_clamped = torch.minimum(lo_clamped, ctrl_hi_grid)
        hi_clamped = torch.maximum(src_hi_grid, ctrl_lo_grid)
        hi_clamped = torch.minimum(hi_clamped, ctrl_hi_grid)

        # Numerical safety: enforce lower <= upper elementwise
        lo_final = torch.minimum(lo_clamped, hi_clamped)
        hi_final = torch.maximum(lo_clamped, hi_clamped)

        out_lower[..., ti, :] = lo_final.to(device_cpu)
        out_upper[..., ti, :] = hi_final.to(device_cpu)

    # Assemble destination payload (preserve useful metadata)
    out_dir = Path('.cache') / 'grid_sets'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_tag}.pkl"
    if out_path.exists() and not force:
        print(f"✓ GridSet already exists: {out_path}")
        return True
    if out_path.exists() and force:
        out_path.unlink(missing_ok=True)

    payload = {
        'tag': out_tag,
        'description': description or (src.get('description', '') + ' [clamped to control limits]'),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'system_name': system_name,
        'input_name': src.get('input_name', ''),
        'set_type': 'box',
        'grid_input_tag': src.get('grid_input_tag'),
        'nn_input_tag': src.get('nn_input_tag'),
        'grid_shape': (*shape, T),
        'state_grid_points': state_axes,
        'time_grid_points': time_axis,
        'box_lower': out_lower,
        'box_upper': out_upper,
    }

    import pickle
    tmp = out_path.with_suffix('.pkl.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out_path)

    widths = (out_upper - out_lower).float()
    print(f"\n✓ Saved clamped GridSet: {out_path}")
    print(f"  Shape: {list(shape) + [T, input_dim]}; Points: {np.prod(shape) * T:,}")
    print(f"  Width stats: mean={float(widths.mean()):.6g}, max={float(widths.max()):.6g}")
    return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python constrain_grid_set.py --grid-set-tag RoverDark_MPC_NN_Box --tag RoverDark_MPC_NN_Box_Clamped
  python constrain_grid_set.py --grid-set-tag my_set --tag my_set_clamped --set description="clamped version"
""",
    )
    p.add_argument('--grid-set-tag', required=True, help='Source GridSet tag to clamp')
    p.add_argument('--tag', required=True, help='Destination GridSet tag (new file)')
    p.add_argument('--force', action='store_true', help='Overwrite destination if it exists')
    
    # Generic override mechanism
    p.add_argument('--set', type=str, action='append', dest='overrides', metavar='KEY=VALUE',
                   help='Override config value (can be repeated)')
    
    # Convenience args
    p.add_argument('--description', default='', help='Description')
    
    args = p.parse_args()
    
    # Build config
    from src.utils.config import parse_key_value_overrides, apply_overrides
    cfg = {'description': args.description}
    if args.overrides:
        set_overrides = parse_key_value_overrides(args.overrides)
        cfg = apply_overrides(cfg, set_overrides)

    ok = clamp_grid_set_to_limits(args.grid_set_tag, args.tag, cfg.get('description', ''), force=bool(args.force))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
