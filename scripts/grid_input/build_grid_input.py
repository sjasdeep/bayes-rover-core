#!/usr/bin/env python3
"""
Build GridInput cache for a specific system-input pair (tagged cache).

Uses grids from config/resolutions.yaml and saves to .cache/grid_inputs/{TAG}.pkl
with metadata and (ideally) pickled system/input objects for reproducibility.

Usage:
  # List available systems and inputs
  python scripts/grid_input/build_grid_input.py --list

  # Build cache for a specific combination
  python scripts/grid_input/build_grid_input.py \
      --system RoverDark \
      --input RoverDark_MPC \
      --tag my_tag \
      --description "short note"
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.inputs import Input
from src.core.systems import System
from src.utils.config import load_resolution_config, parse_key_value_overrides, apply_overrides
from src.utils.registry import (
    get_input_class,
    get_system_class,
    get_system_to_inputs_map,
)


def list_available() -> None:
    mapping = get_system_to_inputs_map()
    print("\n" + "=" * 60)
    print("Available Systems and Inputs")
    print("=" * 60)
    for system_cls, compatible in mapping.items():
        print(f"\n{system_cls.__name__}:")
        if compatible:
            for inp_cls in compatible:
                type_info = f" (type='{getattr(inp_cls, 'type', 'any')}')"
                print(f"  - {inp_cls.__name__}{type_info}")
        else:
            print("  (no compatible inputs)")
    print()


def _require_keys(d: dict, keys):
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Grid config missing required keys: {missing}")
    return d


def build_state_grid_points(system: System, state_resolution: List[int]) -> List[torch.Tensor]:
    if len(state_resolution) != system.state_dim:
        raise ValueError(
            f"State resolution length {len(state_resolution)} does not match system state dimension {system.state_dim}"
        )
    points: List[torch.Tensor] = []
    for i, res in enumerate(state_resolution):
        lo = float(system.state_limits[0, i].item())
        hi = float(system.state_limits[1, i].item())
        if torch.isinf(torch.tensor(lo)) or torch.isinf(torch.tensor(hi)):
            raise ValueError(f"State dimension {i} has infinite limits; cannot build grid")
        points.append(torch.linspace(lo, hi, int(res)))
    return points


def evaluate_on_grid(system: System, input_instance: Input, state_grid_points: List[torch.Tensor], *,
                     time_grid_points: Optional[torch.Tensor]) -> torch.Tensor:
    # Prepare meshgrid
    if time_grid_points is None:
        mesh_grids = torch.meshgrid(*state_grid_points, indexing='ij')
        grid_shape = mesh_grids[0].shape
        state_mesh = torch.stack(mesh_grids, dim=-1)  # [..., state_dim]
        flat_states = state_mesh.reshape(-1, system.state_dim)

        # Batch evaluate (time=0 ignored by time-invariant inputs)
        use_gpu = getattr(input_instance, '_use_gpu', False)
        batch_size = getattr(input_instance, '_batch_size', 100)
        cuda_available = torch.cuda.is_available()
        device = 'cuda' if use_gpu and cuda_available else 'cpu'
        
        # Log attribute sources
        if hasattr(input_instance, '_use_gpu'):
            print(f"use_gpu={use_gpu} (from Input._use_gpu attribute)")
        else:
            print(f"use_gpu={use_gpu} (using default, Input has no _use_gpu attribute)")
        
        if hasattr(input_instance, '_batch_size'):
            print(f"batch_size={batch_size} (from Input._batch_size attribute)")
        else:
            print(f"batch_size={batch_size} (using default, Input has no _batch_size attribute)")
        
        if use_gpu and not cuda_available:
            print(f"device={device} (CUDA not available, falling back to cpu)")
        else:
            print(f"device={device}")
        
        flat_states = flat_states.to(device)

        flat_inputs = []
        total = flat_states.shape[0]
        with tqdm(total=total, desc="Building grid (time-invariant)", unit="pt") as pbar:
            for i in range(0, total, batch_size):
                bs = flat_states[i:i+batch_size]
                out = input_instance.input(bs, 0.0)
                flat_inputs.append(out)
                pbar.update(bs.shape[0])

        flat_inputs = torch.cat(flat_inputs, dim=0)
        flat_inputs = flat_inputs.to('cpu')
        dim = flat_inputs.shape[-1]
        grid_cache = flat_inputs.reshape(list(grid_shape) + [dim])
        return grid_cache

    else:
        # Full state-time grid
        grid_tensors = state_grid_points + [time_grid_points]
        mesh_grids = torch.meshgrid(*grid_tensors, indexing='ij')
        grid_shape = mesh_grids[0].shape
        state_mesh = torch.stack(mesh_grids[:-1], dim=-1)
        time_mesh = mesh_grids[-1]

        flat_states = state_mesh.reshape(-1, system.state_dim)
        flat_times = time_mesh.reshape(-1)

        # Use same GPU/batch settings as time-invariant case
        use_gpu = getattr(input_instance, '_use_gpu', False)
        batch_size = getattr(input_instance, '_batch_size', 100)
        cuda_available = torch.cuda.is_available()
        device = 'cuda' if use_gpu and cuda_available else 'cpu'
        
        # Log attribute sources
        if hasattr(input_instance, '_use_gpu'):
            print(f"use_gpu={use_gpu} (from Input._use_gpu attribute)")
        else:
            print(f"use_gpu={use_gpu} (using default, Input has no _use_gpu attribute)")
        
        if hasattr(input_instance, '_batch_size'):
            print(f"batch_size={batch_size} (from Input._batch_size attribute)")
        else:
            print(f"batch_size={batch_size} (using default, Input has no _batch_size attribute)")
        
        if use_gpu and not cuda_available:
            print(f"device={device} (CUDA not available, falling back to cpu)")
        else:
            print(f"device={device}")
        
        flat_states = flat_states.to(device)
        flat_times = flat_times.to(device)

        flat_inputs = []
        total = flat_states.shape[0]
        with tqdm(total=total, desc="Building grid (time-varying)", unit="pt") as pbar:
            for i in range(0, total, batch_size):
                bs = flat_states[i:i+batch_size]
                bt = flat_times[i:i+batch_size]
                # Group by unique times for efficiency
                uniq_t = torch.unique(bt)
                out_batch = torch.zeros(bs.shape[0], getattr(input_instance, 'dim', 1), 
                                       device=bs.device)
                for t in uniq_t:
                    mask = (bt == t)
                    states_t = bs[mask]
                    out = input_instance.input(states_t, float(t.item()))
                    out_batch[mask] = out
                    pbar.update(states_t.shape[0])
                flat_inputs.append(out_batch)

        flat_inputs = torch.cat(flat_inputs, dim=0)
        flat_inputs = flat_inputs.to('cpu')
        dim = flat_inputs.shape[-1]
        grid_cache = flat_inputs.reshape(list(grid_shape) + [dim])
        return grid_cache


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config-driven usage:
  Grid parameters come from config/resolutions.yaml.
  Use --set KEY=VALUE for ad-hoc overrides.

Examples:
  python build_grid_input.py --system RoverDark --input RoverDark_MPC --tag my_tag
  python build_grid_input.py --system RoverDark --input RoverDark_MPC --tag my_tag --set description="test run"
""",
    )
    parser.add_argument('--list', action='store_true', help='List available systems and inputs')
    parser.add_argument('--system', type=str, help='System class name')
    parser.add_argument('--input', type=str, help='Input class name')
    parser.add_argument('--tag', type=str, help='Tag name for the cache file (required unless --list)')
    parser.add_argument('--config', type=str, default='config/resolutions.yaml', help='Path to grid resolution config')
    parser.add_argument('--force', action='store_true', help='Overwrite existing cache if it exists')
    
    # Generic override mechanism
    parser.add_argument('--set', type=str, action='append', dest='overrides', metavar='KEY=VALUE',
                        help='Override config value (can be repeated). Examples: --set description="my run"')
    
    # Convenience args (shortcuts for --set)
    parser.add_argument('--description', type=str, default='', help='Description (convenience for --set description=X)')

    args = parser.parse_args()

    if args.list:
        list_available()
        return

    if not args.system or not args.input or not args.tag:
        print("Error: --system, --input, and --tag are required")
        sys.exit(1)

    # Load resolution config
    cfg = load_resolution_config(args.system, args.input, args.config)
    cfg = _require_keys(cfg, ['state_resolution', 'time_resolution'])
    
    # Apply CLI convenience args as overrides
    cli_overrides = {}
    if args.description:
        cli_overrides['description'] = args.description
    if cli_overrides:
        cfg = apply_overrides(cfg, cli_overrides)
    
    # Apply --set overrides (highest priority)
    if args.overrides:
        set_overrides = parse_key_value_overrides(args.overrides)
        cfg = apply_overrides(cfg, set_overrides)
        print(f"Applied --set overrides: {set_overrides}")
    
    description = cfg.get('description', '')

    # Check if tag already exists BEFORE doing any computation
    cache_dir = Path('.cache') / 'grid_inputs'
    cache_path = cache_dir / f"{args.tag}.pkl"
    if cache_path.exists():
        if not args.force:
            print(f"\n✗ Tag already exists: {cache_path.name} (refusing to overwrite)")
            print(f"   Path: {cache_path}")
            print(f"   Use --force to overwrite")
            sys.exit(1)
        else:
            print(f"\n⚠ Overwriting existing cache: {cache_path.name}")
            cache_path.unlink()

    # Resolve classes
    system_cls = get_system_class(args.system)
    input_cls = get_input_class(args.input)
    if input_cls is not None and input_cls.__name__ == 'GridInput':
        input_cls = None

    if system_cls is None:
        print(f"Error: System '{args.system}' not found")
        sys.exit(1)
    if input_cls is None:
        print(f"Error: Input '{args.input}' not found")
        sys.exit(1)
    if not issubclass(system_cls, input_cls.system_class):
        print(f"Error: Input '{args.input}' is not compatible with system '{args.system}'")
        sys.exit(1)

    # Instantiate and bind
    system = system_cls()
    input_instance = input_cls()
    input_instance.bind(system)

    # Build grids from config (already loaded above)
    state_grid_points = build_state_grid_points(system, cfg['state_resolution'])
    # Time grid: use only if input is time-varying
    time_grid_points = None
    is_time_invariant = getattr(input_instance, 'time_invariant', False)
    if not is_time_invariant:
        th = float(getattr(system, 'time_horizon'))
        time_grid_points = torch.linspace(0.0, th, int(cfg['time_resolution']))
        print(f"Input is time-varying (Input.time_invariant={is_time_invariant}), building time grid")
    else:
        print(f"Input is time-invariant (Input.time_invariant={is_time_invariant}), skipping time grid")

    print("\nBuilding GridInput cache for:")
    print(f"  System:      {args.system}")
    print(f"  Input:       {args.input}")
    print(f"  Input type:  {getattr(input_instance, 'type', 'any')}")
    print(f"  State res:   {cfg['state_resolution']}")
    if time_grid_points is None:
        print("  Time:        time-invariant")
    else:
        th = float(getattr(system, 'time_horizon'))
        print(f"  Time:        horizon={th}, steps={cfg['time_resolution']}")

    # Evaluate
    grid_cache = evaluate_on_grid(system, input_instance, state_grid_points, time_grid_points=time_grid_points)

    # Prepare cache payload (tag check already done earlier)
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        'tag': args.tag,
        'description': description,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'system_name': args.system,
        'input_name': args.input,
        'input_type': getattr(input_instance, 'type', 'any'),
        # Grid tensors (CPU)
        'state_grid_points': [t.cpu() for t in state_grid_points],
        'time_grid_points': None if time_grid_points is None else time_grid_points.cpu(),
        'grid_cache': grid_cache.cpu(),
        # Also include metadata dict for consumer convenience
        'metadata': {
            'tag': args.tag,
            'description': description,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'system': args.system,
            'input': args.input,
            'input_type': getattr(input_instance, 'type', 'any'),
            'grid_shape': list(grid_cache.shape),
        }
    }

    # Write safely to avoid partial files
    tmp_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)

    # Summary
    total_points = grid_cache.numel() // grid_cache.shape[-1]
    size_mb = grid_cache.element_size() * grid_cache.numel() / (1024**2)
    print("\n✓ Cache built and saved")
    print(f"  File:         {cache_path}")
    print(f"  Grid shape:   {list(grid_cache.shape)}")
    print(f"  Grid points:  {total_points:,}")
    print(f"  Memory (MB):  {size_mb:.2f}")


if __name__ == '__main__':
    main()
