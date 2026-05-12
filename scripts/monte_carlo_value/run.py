#!/usr/bin/env python3
"""
Compute Monte Carlo under-approximate value V_N(x) over a 2D slice.

- Control input is specified like scripts/simulation/simulate.py (by class name and optional tag for cache-backed inputs),
  or read from config/monte_carlo_value.yaml when omitted.
- Disturbance and uncertainty are sampled uniformly at each step within system limits.
- Time horizon and rollout batch size come from the System; time resolution comes from config/resolutions.yaml.
- Slice, grid resolution, total samples, and snapshot schedule come from config/monte_carlo_value.yaml.

Saves cache under .cache/monte_carlo_values/{TAG}.pkl (+ .meta.json).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.utils.registry import instantiate_system
from src.utils.config import load_monte_carlo_config
from src.utils.monte_carlo_value import (
    SliceSpec, compute_monte_carlo_value, save_monte_carlo_cache
)
from src.core.systems import System
from src.core.inputs import Input
from src.utils.cache_loaders import resolve_input_with_class_and_tag


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system', type=str, required=True, help='System class name (e.g., RoverDark)')
    p.add_argument('--control', type=str, required=False, help='Control input class (e.g., RoverDark_MPC, OptimalInputFromValue, GridInput, NNInput). If omitted, falls back to config/monte_carlo_value.yaml')
    p.add_argument('--control-tag', type=str, help='Cache tag when control is one of {OptimalInputFromValue, GridInput, NNInput}. If omitted, falls back to config/monte_carlo_value.yaml')
    p.add_argument('--tag', type=str, required=True, help='Output cache tag (.cache/monte_carlo_values/{tag}.pkl)')
    p.add_argument('--description', type=str, default='', help='Optional description')
    p.add_argument('--mc-config', type=str, default='config/monte_carlo_value.yaml', help='Path to Monte Carlo value config')
    p.add_argument('--preset', type=str, default=None, help='Optional preset name under the system in monte_carlo.yaml')
    p.add_argument('--time-horizon', type=float, default=None, help='Override System.time_horizon for this run (seconds)')
    return p.parse_args()


def _resolve_control(system: System, name: str, tag: str | None, total_sim_horizon: float) -> tuple[Input, str]:
    """Resolve control input using consolidated helper."""
    ctrl = resolve_input_with_class_and_tag(
        system,
        input_class=name,
        tag=tag,
        role='control',
    )
    return ctrl, name


def main():
    args = parse_args()

    # System and device
    try:
        system = instantiate_system(args.system)
    except ValueError as e:
        raise SystemExit(str(e))
    # Optional override of time horizon
    if args.time_horizon is not None:
        try:
            system.time_horizon = float(args.time_horizon)
            print(f"Time horizon override: {system.time_horizon} (from CLI --time-horizon)")
        except Exception as e:
            raise SystemExit(f"Failed to set system.time_horizon: {e}")
    use_gpu = getattr(system, '_use_gpu', False) and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"Device: {device} | CUDA available={torch.cuda.is_available()} | system._use_gpu={getattr(system, '_use_gpu', False)}")

    # Time horizon from system; dt from Monte Carlo config
    H = float(getattr(system, 'time_horizon'))

    # Monte Carlo config (slice/grid/samples/snapshots and optional control defaults)
    mc = load_monte_carlo_config(args.system, preset_name=args.preset, path=args.mc_config)
    if not mc:
        raise SystemExit(f"No Monte Carlo config found for system '{args.system}' in {args.mc_config}")

    # Determine control from CLI or config
    cfg_control = mc.get('control')
    cfg_control_tag = mc.get('control-tag') or mc.get('control_tag')
    chosen_control = args.control or cfg_control
    chosen_control_tag = args.control_tag or cfg_control_tag
    if chosen_control is None:
        raise SystemExit("--control not provided and no 'control' entry found in config; please specify a control class.")
    if chosen_control in essential_inputs and not chosen_control_tag:
        raise SystemExit("--control-tag not provided and no 'control-tag' in config; required for control in {GridInput, OptimalInputFromValue, NNInput}.")

    # Control
    source = 'CLI' if args.control else 'config'
    control, control_name = _resolve_control(system, chosen_control, chosen_control_tag, H)
    print(f"Control selected: {control_name} (from {source})" + (f", tag={chosen_control_tag}" if chosen_control_tag else ""))
    # Move control/cache to device if supported (e.g., GridInput has .to)
    if hasattr(control, 'to'):
        try:
            control.to(device)
        except TypeError:
            control.to(device=device)  # accommodate signatures expecting kwarg

    slices = mc.get('slices', {})
    vary = slices.get('vary_dims', [0, 1])
    fixed = slices.get('fixed', {})
    slice_spec = SliceSpec(vary_dims=(int(vary[0]), int(vary[1])), fixed={int(k): float(v) for k, v in fixed.items()})
    grid_res = tuple(int(x) for x in mc.get('grid_resolution', [101, 101]))
    dt = float(mc.get('dt', 0.05))
    total_samples = int(mc.get('total_samples_per_state', 100))
    snapshot_samples = [int(s) for s in mc.get('snapshot_samples', [10, 20, 50, 100])]

    # Compute
    payload = compute_monte_carlo_value(
        system=system,
        control=control,
        slice_spec=slice_spec,
        grid_resolution=(grid_res[0], grid_res[1]),
        total_samples_per_state=total_samples,
        snapshot_samples=snapshot_samples,
        dt=dt,
        device=device,
    )

    # Save cache
    save_monte_carlo_cache(
        tag=args.tag,
        system=system,
        control_name=control_name,
        description=args.description or '',
        payload=payload,
    )

    cache_path = Path('.cache') / 'monte_carlo_values' / f'{args.tag}.pkl'
    print(f"\n✓ Monte Carlo value saved: {cache_path}")
    print(f"  System:    {args.system}")
    print(f"  Control:   {control_name}")
    print(f"  Time H:    {H:.3f}s  |  dt={dt}")
    print(f"  Grid:      {payload['meta']['shape2d']}")
    print(f"  Snapshots: {len(payload['snapshots'])} at {snapshot_samples}")


if __name__ == '__main__':
    main()
